import math, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, pytest
from inventory_core import inventory as I, simulation as S, optimization as O, data as D, forecasting as F

def test_z_and_formula_hand_calc():
    z = I.z_value(0.95)
    assert z == pytest.approx(1.6449, abs=1e-4)
    # SS = 1.645*sqrt(20^2*4 + 100^2*1^2) = 1.645*sqrt(1600+10000)
    assert I.safety_stock(z, 20, 4, 100, 1) == pytest.approx(z * math.sqrt(11600))
    assert I.reorder_point(100, 4, 50) == 450

def test_formula_rejects_bad_inputs():
    with pytest.raises(ValueError): I.z_value(1.0)
    with pytest.raises(ValueError): I.safety_stock(1.6, -1, 2, 10, 1)

def test_classification_and_order_qty():
    assert I.classify(40, 50, 120, 10) == "Critical Stockout Risk"
    assert I.classify(100, 50, 120, 10) == "Reorder Now"
    assert I.classify(130, 50, 120, 10) == "Monitor"
    assert I.classify(500, 50, 120, 10) == "Healthy"
    assert I.order_quantity(100, 120, 50) == 50 and I.order_quantity(10, 120, 50) == 150
    assert I.order_quantity(200, 120, 50) == 0

def test_monte_carlo_reproducible_and_matches_normal_theory():
    a = S.lead_time_demand(20000, 100, 20, 4, 0.0, "Normal (truncated at 0)", "Fixed", seed=7)
    b = S.lead_time_demand(20000, 100, 20, 4, 0.0, "Normal (truncated at 0)", "Fixed", seed=7)
    assert np.array_equal(a["dlt"], b["dlt"])
    r = S.summarise_dlt(a["dlt"], 0.95, d_mean=100, l_mean=4)
    ss_formula = I.safety_stock(I.z_value(0.95), 20, 4, 100, 0.0)
    assert r["ss_mc"] == pytest.approx(ss_formula, rel=0.05)     # normal, fixed L -> formula is exact

def test_dependence_increases_buffer():
    ind = S.summarise_dlt(S.lead_time_demand(20000, 100, 30, 3, 1, rho=0.0, seed=1)["dlt"], .95, d_mean=100, l_mean=3)
    dep = S.summarise_dlt(S.lead_time_demand(20000, 100, 30, 3, 1, rho=0.6, seed=1)["dlt"], .95, d_mean=100, l_mean=3)
    assert dep["ss_mc"] > ind["ss_mc"]

def test_policy_sim_and_optimiser_respect_constraints():
    sc = O.Scenario(d_mean=100, d_sd=30, l_mean=2, l_sd=0.5, unit_cost=5, shortage_cost=5,
                    target_csl=0.95, moq=200, n_paths=100)
    res = O.optimise(sc, n_trials=40)
    assert res["any_feasible"]
    b = res["best"]
    assert b["q"] >= 200 and b["cycle_service_level"] >= 0.95
    h = res["history"]; feas = h[h.feasible]
    assert b["cost"] == pytest.approx(feas.cost.min())
    r1 = O.evaluate(b["rop"], b["q"], sc, S.make_paths(50, 60, 100, 30, 2, 0.5, seed=3))
    r2 = O.evaluate(b["rop"], b["q"], sc, S.make_paths(50, 60, 100, 30, 2, 0.5, seed=3))
    assert r1["total_cost"] == r2["total_cost"]

def test_validation_handles_bad_rows_and_missing_fields():
    df = pd.DataFrame({"InvoiceDate": ["2011-12-05", "not a date", "2011-12-06", "2011-12-06"],
                       "StockCode": ["84077", "84077", "84077", "84077"], "Quantity": [10, 5, "x", 3]})
    bad = D.validate_new_data(df, {"date": "InvoiceDate", "sku": "StockCode"})
    assert not bad.ok
    m = D.detect_columns(df)
    assert m == {"date": "InvoiceDate", "sku": "StockCode", "quantity": "Quantity"}
    res = D.validate_new_data(df, m)
    assert res.ok and len(res.warnings) >= 2 and res.data["demand"].sum() == 13

def test_append_reconciles_overlaps():
    weeks = pd.date_range("2011-01-03", periods=4, freq="W-MON")
    panel = pd.DataFrame({"week": weeks, "sku": "A", "description": "x", "demand": [1., 2, 3, 4],
                          "avg_price": 1.0, "price_cut": 0, "holidays": 0})
    new = pd.DataFrame({"sku": ["A", "A", "ZZZ"], "week": [weeks[-1], weeks[-1] + pd.Timedelta(weeks=1), weeks[0]],
                        "demand": [40., 50., 9.], "avg_price": [1.0, 1.0, 1.0]})
    out, info = D.append_weekly(panel, new)
    assert info["replaced"] == 1 and info["new_weeks"] == 1 and "ZZZ" in info["unknown_skus"]
    assert out["demand"].tolist() == [1, 2, 3, 40, 50]

def test_forecasting_models_no_lookahead():
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2010-01-04", periods=80, freq="W-MON")
    rows = []
    for s in ["A", "B"]:
        y = 50 + 10 * np.sin(np.arange(80) / 8) + rng.normal(0, 3, 80)
        rows.append(pd.DataFrame({"week": weeks, "sku": s, "description": s, "demand": y,
                                  "avg_price": 2.0, "price_cut": 0, "holidays": 0}))
    panel = pd.concat(rows)
    bt = F.backtest(panel, [60], horizon=4)
    assert set(bt.model) == set(F.MODEL_NAMES) and (bt.week > weeks[60]).all()
    m = F.metrics(bt)
    assert (m.WAPE >= 0).all() and m.loc[m.model == "Naive", "n"].iat[0] == 8
