"""Analytical inventory formulas, status classification and recommendations.

Units: every quantity is per WEEK (demand per week, lead time in weeks), so the formula
inputs are always in the same time unit.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

STATUS_ORDER = ["Critical Stockout Risk", "Reorder Now", "Monitor", "Healthy"]
STATUS_COLORS = {"Healthy": "#14b8a6", "Monitor": "#f59e0b", "Reorder Now": "#f97316",
                 "Critical Stockout Risk": "#ef4444", "No data": "#64748b"}


def z_value(service_level: float) -> float:
    """Service-level factor Z for a cycle service level (probability of no stockout per cycle)."""
    if not 0.5 <= service_level < 1:
        raise ValueError("service level must be in [0.5, 1)")
    return float(norm.ppf(service_level))


def safety_stock(z: float, sigma_d: float, lead_time: float, d_bar: float, sigma_l: float) -> float:
    """SS = Z * sqrt(sigma_D^2 * L + D_bar^2 * sigma_L^2)  (lecture formula).
    Assumes weekly demands are i.i.d., lead time is independent of demand, and lead-time
    demand is approximately normal."""
    if min(sigma_d, lead_time, d_bar, sigma_l) < 0:
        raise ValueError("inputs must be non-negative")
    return float(z * math.sqrt(sigma_d ** 2 * lead_time + d_bar ** 2 * sigma_l ** 2))


def lead_time_demand_sd(sigma_d, lead_time, d_bar, sigma_l) -> float:
    return float(math.sqrt(sigma_d ** 2 * lead_time + d_bar ** 2 * sigma_l ** 2))


def reorder_point(d_bar: float, lead_time: float, ss: float) -> float:
    """ROP = D_bar * L + SS."""
    return float(d_bar * lead_time + ss)


def eoq(annual_demand: float, order_cost: float, holding_cost_unit_year: float) -> float:
    if holding_cost_unit_year <= 0:
        raise ValueError("holding cost must be positive")
    return float(math.sqrt(2 * annual_demand * order_cost / holding_cost_unit_year))


def inventory_position(on_hand: float, on_order: float = 0.0, backorders: float = 0.0) -> float:
    return float(on_hand + on_order - backorders)


def classify(ip: float, ss: float, rop: float, weekly_demand: float) -> str:
    """Healthy / Monitor / Reorder Now / Critical Stockout Risk from the inventory position."""
    if ip <= ss:
        return "Critical Stockout Risk"
    if ip <= rop:
        return "Reorder Now"
    if ip <= rop + 2 * weekly_demand:          # within two weeks of demand above the ROP
        return "Monitor"
    return "Healthy"


def order_quantity(ip: float, rop: float, q_base: float, moq: float = 0.0) -> float:
    """(s, nQ) rule: when IP <= ROP, order the smallest multiple of Q that lifts IP above ROP."""
    if ip > rop:
        return 0.0
    q = max(q_base, moq)
    n = math.floor((rop - ip) / q) + 1
    return float(n * q)


ACTIONS = {
    "Critical Stockout Risk": "Expedite: place an order now and consider express shipping",
    "Reorder Now": "Place a replenishment order this week",
    "Monitor": "No order yet; review again next week",
    "Healthy": "No action needed",
}


def sku_policy_table(params: pd.DataFrame, service_level: float, lead_mean: float, lead_sd: float,
                     order_cost: float, holding_rate: float, moq: float = 0.0) -> pd.DataFrame:
    """params needs: sku, description, d_bar, sigma_d, unit_cost, on_hand, on_order, backorders.
    Returns SS, ROP, EOQ, inventory position, status, order quantity and action per SKU."""
    z = z_value(service_level)
    t = params.copy()
    t["safety_stock"] = [safety_stock(z, s, lead_mean, d, lead_sd) for d, s in zip(t["d_bar"], t["sigma_d"])]
    t["rop"] = t["d_bar"] * lead_mean + t["safety_stock"]
    t["eoq"] = [max(eoq(d * 52, order_cost, c * holding_rate), moq) for d, c in zip(t["d_bar"], t["unit_cost"])]
    t["inventory_position"] = t["on_hand"] + t["on_order"] - t["backorders"]
    t["weeks_of_cover"] = t["inventory_position"] / t["d_bar"].replace(0, np.nan)
    t["status"] = [classify(ip, ss, r, d) for ip, ss, r, d in
                   zip(t["inventory_position"], t["safety_stock"], t["rop"], t["d_bar"])]
    t["order_qty"] = [order_quantity(ip, r, q, moq) for ip, r, q in zip(t["inventory_position"], t["rop"], t["eoq"])]
    t["action"] = t["status"].map(ACTIONS)
    return t
