"""Bayesian optimisation of the (s, nQ) reorder policy with Optuna.

Decision variables : reorder point s (ROP) and order quantity Q.
Objective          : minimise expected annual cost = holding + ordering + shortage,
                     estimated by simulate_policy() on a FIXED set of demand / lead-time
                     paths (common random numbers), so every candidate is compared fairly.
Constraints        : cycle service level >= target, 99th-percentile peak stock <= warehouse
                     capacity, average stock value <= inventory budget, Q >= MOQ (bound).
Sampler            : TPE (tree-structured Parzen estimator, a sequential model-based Bayesian
                     optimiser) or a Gaussian-process sampler. Optuna treats a trial as feasible
                     when every constraint value is <= 0.
The chosen policy is then re-evaluated on an independent set of paths (new seed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .inventory import eoq, reorder_point, safety_stock, z_value
from .simulation import make_paths, simulate_policy


@dataclass
class Scenario:
    d_mean: float
    d_sd: float
    l_mean: float
    l_sd: float
    unit_cost: float
    holding_rate: float = 0.25          # per year, share of unit cost
    order_cost: float = 50.0            # per order
    shortage_cost: float = 5.0          # per unit short
    target_csl: float = 0.95
    capacity: float = 1e9               # units
    budget: float = 1e12                # average stock value
    moq: float = 1.0
    q_max: float | None = None
    demand_dist: str = "Gamma"
    lead_dist: str = "Gamma"
    empirical: tuple | None = None
    n_paths: int = 200
    weeks: int = 60
    seed: int = 42


def baseline_policy(sc: Scenario) -> dict:
    """Textbook policy: ROP from the lecture formula at the target service level; Q = EOQ."""
    z = z_value(sc.target_csl)
    ss = safety_stock(z, sc.d_sd, sc.l_mean, sc.d_mean, sc.l_sd)
    q = max(eoq(sc.d_mean * 52, sc.order_cost, sc.unit_cost * sc.holding_rate), sc.moq)
    if sc.q_max:
        q = min(q, sc.q_max)
    return {"rop": reorder_point(sc.d_mean, sc.l_mean, ss), "q": q, "ss": ss}


def _constraints(res: dict, sc: Scenario) -> list[float]:
    return [sc.target_csl - res["cycle_service_level"],
            (res["p99_max_on_hand"] - sc.capacity) / max(sc.capacity, 1),
            (res["avg_on_hand"] * sc.unit_cost - sc.budget) / max(sc.budget, 1)]


def evaluate(rop, q, sc: Scenario, paths) -> dict:
    res = simulate_policy(rop, q, paths, sc.unit_cost, sc.holding_rate, sc.order_cost, sc.shortage_cost)
    cons = _constraints(res, sc)
    res.update(rop=rop, q=q, ss=rop - sc.d_mean * sc.l_mean, feasible=all(c <= 1e-12 for c in cons),
               csl_gap=cons[0], capacity_gap=cons[1], budget_gap=cons[2])
    return res


def optimise(sc: Scenario, n_trials: int = 60, sampler: str = "TPE", callback=None) -> dict:
    import warnings
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", message=".*constraints_func.*")   # Optuna 5 deprecation notice
    paths = make_paths(sc.n_paths, sc.weeks, sc.d_mean, sc.d_sd, sc.l_mean, sc.l_sd,
                       sc.demand_dist, sc.lead_dist, sc.empirical, sc.seed)
    base = baseline_policy(sc)
    mu_lt = sc.d_mean * sc.l_mean
    rop_lo, rop_hi = 0.5 * mu_lt, max(3.0 * base["rop"], mu_lt + 4 * sc.d_sd * math.sqrt(sc.l_mean + 1))
    q_lo = max(sc.moq, 1.0)
    q_hi = sc.q_max or max(6 * base["q"], 26 * sc.d_mean, q_lo + 1)

    def constraints_func(trial):
        return trial.user_attrs["constraints"]

    if sampler == "GP":
        s = optuna.samplers.GPSampler(seed=sc.seed, constraints_func=constraints_func, n_startup_trials=10)
    else:
        s = optuna.samplers.TPESampler(seed=sc.seed, constraints_func=constraints_func,
                                       n_startup_trials=10, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=s)
    # warm start with the textbook policy so the optimiser starts from a sensible point
    study.enqueue_trial({"rop": float(np.clip(base["rop"], rop_lo, rop_hi)),
                         "q": float(np.clip(base["q"], q_lo, q_hi))})

    def objective(trial):
        rop = trial.suggest_float("rop", rop_lo, rop_hi)
        q = trial.suggest_float("q", q_lo, q_hi)
        r = evaluate(rop, q, sc, paths)
        trial.set_user_attr("constraints", [r["csl_gap"], r["capacity_gap"], r["budget_gap"]])
        for k in ("cycle_service_level", "fill_rate", "holding_cost", "ordering_cost", "shortage_cost",
                  "p99_max_on_hand", "avg_on_hand", "feasible"):
            trial.set_user_attr(k, r[k])
        return r["total_cost"]

    cbs = [callback] if callback else None
    study.optimize(objective, n_trials=n_trials, callbacks=cbs)
    hist = pd.DataFrame([{"trial": t.number, "rop": t.params["rop"], "q": t.params["q"], "cost": t.value,
                          **{k: v for k, v in t.user_attrs.items() if k != "constraints"}}
                         for t in study.trials])
    hist["best_feasible_so_far"] = hist["cost"].where(hist["feasible"]).cummin().ffill()
    feas = hist[hist["feasible"]]
    if feas.empty:
        best_row = hist.sort_values("cycle_service_level", ascending=False).iloc[0]
    else:
        best_row = feas.loc[feas["cost"].idxmin()]
    # independent validation with fresh random paths
    vpaths = make_paths(sc.n_paths * 2, sc.weeks, sc.d_mean, sc.d_sd, sc.l_mean, sc.l_sd,
                        sc.demand_dist, sc.lead_dist, sc.empirical, sc.seed + 1000)
    best_val = evaluate(best_row["rop"], best_row["q"], sc, vpaths)
    base_opt = evaluate(base["rop"], base["q"], sc, paths)
    base_val = evaluate(base["rop"], base["q"], sc, vpaths)
    return {"history": hist, "best": best_row.to_dict(), "best_validation": best_val,
            "baseline": base, "baseline_eval": base_opt, "baseline_validation": base_val,
            "bounds": {"rop": (rop_lo, rop_hi), "q": (q_lo, q_hi)}, "scenario": asdict(sc),
            "any_feasible": not feas.empty, "sampler": sampler, "n_trials": n_trials}
