"""Monte Carlo engine: lead-time demand, risk scenarios and (s, nQ) policy simulation.

Demand and lead time are drawn through a Gaussian copula, so their dependence can be set
with `rho` (rho = 0 reproduces the independence assumed by the analytical formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DEMAND_DISTS = ["Gamma", "Normal (truncated at 0)", "Lognormal", "Empirical bootstrap"]
LEAD_DISTS = ["Gamma", "Normal (truncated at 0)", "Uniform", "Fixed"]


def _demand_ppf(u, dist, mean, sd, empirical=None):
    sd = max(sd, 1e-9)
    if dist == "Gamma":
        k = (mean / sd) ** 2
        return stats.gamma.ppf(u, a=k, scale=mean / k)
    if dist == "Normal (truncated at 0)":
        return np.maximum(stats.norm.ppf(u, mean, sd), 0)
    if dist == "Lognormal":
        s2 = np.log(1 + (sd / mean) ** 2)
        return stats.lognorm.ppf(u, s=np.sqrt(s2), scale=mean * np.exp(-s2 / 2))
    if dist == "Empirical bootstrap":
        e = np.asarray(empirical, dtype=float)
        e = e * (mean / e.mean()) if e.mean() > 0 else e      # rescale to the scenario mean
        return np.quantile(e, u, method="inverted_cdf")
    raise ValueError(dist)


def _lead_ppf(u, dist, mean, sd):
    if dist == "Fixed" or sd <= 0:
        return np.full_like(u, mean, dtype=float)
    if dist == "Gamma":
        k = (mean / sd) ** 2
        return stats.gamma.ppf(u, a=k, scale=mean / k)
    if dist == "Normal (truncated at 0)":
        return np.maximum(stats.norm.ppf(u, mean, sd), 0.05)
    if dist == "Uniform":
        half = np.sqrt(3) * sd                         # same mean and SD as requested
        return np.maximum(mean - half + 2 * half * u, 0.05)
    raise ValueError(dist)


def lead_time_demand(n: int, d_mean: float, d_sd: float, l_mean: float, l_sd: float,
                     demand_dist: str = "Gamma", lead_dist: str = "Gamma", rho: float = 0.0,
                     empirical=None, seed: int = 42) -> dict:
    """Simulate n replenishment cycles. Lead time L is continuous (weeks); demand over the lead
    time sums ceil(L) weekly draws, with the last week weighted by the fractional part."""
    rng = np.random.default_rng(seed)
    zl = rng.standard_normal(n)
    L = _lead_ppf(stats.norm.cdf(zl), lead_dist, l_mean, l_sd)
    lmax = int(np.ceil(L.max()))
    e = rng.standard_normal((n, lmax))
    zd = rho * zl[:, None] + np.sqrt(1 - rho ** 2) * e        # weekly demand correlated with L
    D = _demand_ppf(stats.norm.cdf(zd), demand_dist, d_mean, d_sd, empirical)
    w = np.clip(L[:, None] - np.arange(lmax)[None, :], 0, 1)
    dlt = (D * w).sum(axis=1)
    return {"dlt": dlt, "lead_times": L, "weekly": D[:, 0], "seed": seed, "n": n}


def summarise_dlt(dlt: np.ndarray, service_level: float, rop: float | None = None, q: float | None = None,
                  d_mean: float | None = None, l_mean: float | None = None) -> dict:
    """Safety stock and risk metrics from simulated lead-time demand."""
    q_sl = float(np.quantile(dlt, service_level))
    expected = float(d_mean * l_mean) if d_mean is not None else float(dlt.mean())
    out = {"mean_dlt": float(dlt.mean()), "sd_dlt": float(dlt.std(ddof=1)),
           "rop_mc": q_sl, "ss_mc": q_sl - expected, "p95": float(np.quantile(dlt, .95)),
           "p99": float(np.quantile(dlt, .99))}
    if rop is not None:
        short = np.clip(dlt - rop, 0, None)
        out["stockout_prob"] = float((dlt > rop).mean())               # 1 - cycle service level
        out["cycle_service_level"] = 1 - out["stockout_prob"]
        out["expected_shortage"] = float(short.mean())                   # units short per cycle
        if q:
            out["fill_rate"] = float(1 - short.mean() / q)                # share of demand met from stock
    return out


SCENARIOS = {
    "Normal": dict(d_mult=1.0, dsd_mult=1.0, l_add=0.0, lsd_mult=1.0),
    "Demand spike (+30% mean)": dict(d_mult=1.3, dsd_mult=1.3, l_add=0.0, lsd_mult=1.0),
    "Delayed supplier (+1 wk, 2x var.)": dict(d_mult=1.0, dsd_mult=1.0, l_add=1.0, lsd_mult=2.0),
    "High variability (1.5x SD)": dict(d_mult=1.0, dsd_mult=1.5, l_add=0.0, lsd_mult=1.0),
}


def scenario_table(n, d_mean, d_sd, l_mean, l_sd, service_level, rop, demand_dist="Gamma",
                   lead_dist="Gamma", rho=0.0, empirical=None, seed=42) -> pd.DataFrame:
    """Analytical vs Monte Carlo buffer under each risk scenario, plus the risk of keeping the
    current ROP unchanged. Same seed for every scenario (common random numbers)."""
    from .inventory import safety_stock, z_value
    z = z_value(service_level)
    rows = []
    for name, s in SCENARIOS.items():
        dm, ds = d_mean * s["d_mult"], d_sd * s["dsd_mult"]
        lm, ls = l_mean + s["l_add"], l_sd * s["lsd_mult"]
        sim = lead_time_demand(n, dm, ds, lm, ls, demand_dist, lead_dist, rho, empirical, seed)
        r = summarise_dlt(sim["dlt"], service_level, rop=rop, d_mean=dm, l_mean=lm)
        rows.append({"scenario": name, "mean demand/wk": dm, "demand SD/wk": ds, "lead time (wk)": lm,
                     "lead-time SD (wk)": ls, "SS formula": safety_stock(z, ds, lm, dm, ls),
                     "SS Monte Carlo": r["ss_mc"], "ROP Monte Carlo": r["rop_mc"],
                     "P(stockout) at current ROP": r["stockout_prob"],
                     "Exp. shortage/cycle at current ROP": r["expected_shortage"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- policy simulation
def make_paths(n_paths: int, weeks: int, d_mean, d_sd, l_mean, l_sd, demand_dist="Gamma",
               lead_dist="Gamma", empirical=None, seed=42) -> dict:
    """Common random numbers for policy comparison: demand per week and the lead time that an
    order placed in that week would get (rounded to whole weeks, minimum 1)."""
    rng = np.random.default_rng(seed)
    D = _demand_ppf(rng.random((n_paths, weeks)), demand_dist, d_mean, d_sd, empirical)
    L = np.maximum(np.rint(_lead_ppf(rng.random((n_paths, weeks)), lead_dist, l_mean, l_sd)), 1).astype(int)
    return {"D": D, "L": L, "seed": seed}


def simulate_policy(rop: float, q: float, paths: dict, unit_cost: float, holding_rate: float,
                    order_cost: float, shortage_cost: float, warmup: int = 8,
                    init_on_hand: float | None = None) -> dict:
    """Weekly (s, nQ) continuous-review simulation with backorders.
    Costs: holding = unit_cost*holding_rate/52 per unit per week on hand; ordering = order_cost
    per order; shortage = shortage_cost per unit of demand not met from stock.
    Returns annualised cost components and service metrics (averaged over paths)."""
    D, Lt = paths["D"], paths["L"]
    n, T = D.shape
    h = unit_cost * holding_rate / 52
    pipe = np.zeros((n, T + int(Lt.max()) + 1))
    net = np.full(n, float(init_on_hand if init_on_hand is not None else rop + q))
    on_order = np.zeros(n)
    hold = np.zeros(n); orders = np.zeros(n); short = np.zeros(n); dem = np.zeros(n)
    cycles = np.zeros(n); bad_cycles = np.zeros(n); flag = np.zeros(n, bool)
    max_oh = np.zeros(n); oh_sum = np.zeros(n)
    rows = np.arange(n)
    for t in range(T):
        arr = pipe[:, t]
        net += arr
        on_order -= arr
        counted = t >= warmup
        if counted:
            arrived = arr > 0
            cycles += arrived
            bad_cycles += arrived & flag
            flag[arrived] = False
            max_oh = np.maximum(max_oh, np.maximum(net, 0))
        before = net.copy()
        net -= D[:, t]
        new_short = np.maximum(-net, 0) - np.maximum(-before, 0)
        if counted:
            short += new_short
            dem += D[:, t]
            hold += h * np.maximum(net, 0)
            oh_sum += np.maximum(net, 0)
            flag |= net < 0
        ip = net + on_order
        need = ip <= rop
        if need.any():
            k = np.floor((rop - ip[need]) / q) + 1
            qty = k * q
            lt = Lt[need, t]
            pipe[rows[need], t + lt] += qty
            on_order[need] += qty
            if counted:
                orders[need] += 1
    weeks = T - warmup
    scale = 52 / weeks
    holding = hold.mean() * scale
    ordering = orders.mean() * order_cost * scale
    shortage = short.mean() * shortage_cost * scale
    csl = 1 - bad_cycles.sum() / max(cycles.sum(), 1)
    return {
        "total_cost": holding + ordering + shortage, "holding_cost": holding, "ordering_cost": ordering,
        "shortage_cost": shortage, "cycle_service_level": float(csl),
        "fill_rate": float(1 - short.sum() / max(dem.sum(), 1e-9)),
        "avg_on_hand": float(oh_sum.mean() / weeks), "p99_max_on_hand": float(np.quantile(max_oh, .99)),
        "orders_per_year": float(orders.mean() * scale),
    }
