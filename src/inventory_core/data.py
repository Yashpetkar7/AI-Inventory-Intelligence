"""Data loading, cleaning, validation and weekly demand features.

Data lineage used throughout the project:
  OBSERVED  - columns that come straight from the UCI Online Retail II transactions
  DERIVED   - computed from observed data (weekly demand, average price, price-cut flag)
  EXTERNAL  - public calendar data (UK bank holidays from the `holidays` package)
  ASSUMED   - user-configured scenario parameters (lead time, costs, stock on hand)
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/allanvc/onlineretail2/master/data/onlineretail2.rda"
DATA_SHA256 = "be2480b1fcb1fa123f90d731f63ab3cf44f6c97cba0a38eabee93759d411c6f7"
FIRST_FULL_WEEK = pd.Timestamp("2009-12-07")   # first Monday after the data starts (Tue 1 Dec 2009)
END_EXCLUSIVE = pd.Timestamp("2011-12-05")     # data ends Fri 9 Dec 2011, so that week is partial

REQUIRED_FIELDS = ["date", "sku", "quantity"]
OPTIONAL_FIELDS = ["price", "description", "invoice", "location", "promotion"]
COLUMN_ALIASES = {
    "date": ["invoicedate", "date", "order_date", "orderdate", "ds", "timestamp", "week"],
    "sku": ["stockcode", "sku", "product_id", "productid", "item", "item_id", "product"],
    "quantity": ["quantity", "qty", "demand", "units", "sales", "y"],
    "price": ["price", "unitprice", "unit_price", "sell_price"],
    "description": ["description", "product_name", "name"],
    "invoice": ["invoice", "invoiceno", "order_id"],
    "location": ["country", "store", "store_id", "location"],
    "promotion": ["promotion", "promo", "on_promotion"],
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_raw(path_or_bytes) -> pd.DataFrame:
    """Read the Online Retail II R data file (.rda) or a CSV with the same columns."""
    import pyreadr
    if isinstance(path_or_bytes, (bytes, bytearray)):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".rda", delete=False) as f:
            f.write(path_or_bytes)
            tmp = f.name
        try:
            df = pyreadr.read_r(tmp)["onlineretail2"]
        finally:
            os.remove(tmp)
    elif str(path_or_bytes).endswith(".rda"):
        df = pyreadr.read_r(str(path_or_bytes))["onlineretail2"]
    else:
        df = pd.read_csv(path_or_bytes)
    return df.rename(columns={"Customer ID": "CustomerID"})


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep genuine product sales; return the cleaned frame and a data-quality report."""
    rep = {"rows_raw": int(len(raw))}
    df = raw.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.upper().str.strip()
    rep["missing_by_column"] = {k: int(v) for k, v in df.isna().sum().items()}
    rep["exact_duplicates"] = int(df.duplicated().sum())
    cancelled = df["Invoice"].str.startswith("C")
    rep["cancellation_rows"] = int(cancelled.sum())
    nonpos = (df["Quantity"] <= 0) | (df["Price"] <= 0)
    rep["non_positive_qty_or_price_rows"] = int((nonpos & ~cancelled).sum())
    product = df["StockCode"].str.match(r"^\d{5}[A-Z]?$")
    rep["non_product_code_rows"] = int((~product & ~cancelled & ~nonpos).sum())  # POST, DOT, M, fees ...
    df = df[~cancelled & ~nonpos & product].drop_duplicates()
    rep["rows_clean"] = int(len(df))
    rep["skus_clean"] = int(df["StockCode"].nunique())
    rep["date_min"] = str(df["InvoiceDate"].min())
    rep["date_max"] = str(df["InvoiceDate"].max())
    return df, rep


def uk_holidays_per_week(weeks: pd.DatetimeIndex) -> pd.Series:
    """EXTERNAL factor: number of England bank holidays falling in each Monday-Sunday week."""
    import holidays
    years = sorted({d.year for d in weeks} | {d.year + 1 for d in weeks})
    hol = holidays.country_holidays("GB", subdiv="ENG", years=years)
    hd = pd.to_datetime(list(hol.keys()))
    wk = hd.to_period("W-SUN").start_time
    counts = pd.Series(1, index=wk).groupby(level=0).sum()
    return counts.reindex(weeks, fill_value=0).astype(int)


def weekly_panel(clean: pd.DataFrame, n_skus: int = 24, min_active_share: float = 0.9) -> pd.DataFrame:
    """Weekly demand per SKU for the top SKUs that sell in >= min_active_share of weeks.

    Returns a long frame: week, sku, description, demand (OBSERVED sum of units),
    avg_price (DERIVED), price_cut (DERIVED proxy), holidays (EXTERNAL).
    """
    df = clean.copy()
    df["week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    df = df[(df["week"] >= FIRST_FULL_WEEK) & (df["week"] < END_EXCLUSIVE)]
    df["revenue"] = df["Quantity"] * df["Price"]
    grid = df.groupby(["StockCode", "week"]).agg(demand=("Quantity", "sum"), revenue=("revenue", "sum"))
    weeks = pd.date_range(FIRST_FULL_WEEK, END_EXCLUSIVE - pd.Timedelta(days=7), freq="W-MON")
    wide = grid["demand"].unstack(fill_value=0).reindex(columns=weeks, fill_value=0)
    active = (wide > 0).mean(axis=1)
    chosen = wide[active >= min_active_share].sum(axis=1).sort_values(ascending=False).index[:n_skus]
    desc = (df[df["StockCode"].isin(chosen)].groupby("StockCode")["Description"]
            .agg(lambda s: s.mode().iat[0] if len(s.mode()) else str(s.iloc[0])))
    idx = pd.MultiIndex.from_product([chosen, weeks], names=["sku", "week"])
    g = grid.reindex(idx).fillna(0.0).reset_index()
    g["avg_price"] = np.where(g["demand"] > 0, g["revenue"] / g["demand"].replace(0, np.nan), np.nan)
    g["avg_price"] = g.groupby("sku")["avg_price"].transform(lambda s: s.ffill().bfill())
    med = g.groupby("sku")["avg_price"].transform("median")
    g["price_cut"] = (g["avg_price"] < 0.9 * med).astype(int)
    g["holidays"] = uk_holidays_per_week(pd.DatetimeIndex(g["week"])).values
    g["description"] = g["sku"].map(desc).str.strip().str.title()
    return g[["week", "sku", "description", "demand", "avg_price", "price_cut", "holidays"]]


# ---------------------------------------------------------------- uploads / new data
def detect_columns(df: pd.DataFrame) -> dict:
    """Suggest a mapping {field: column} by matching common column names."""
    norm = {c: c.lower().replace(" ", "").replace("-", "_") for c in df.columns}
    out = {}
    for field, aliases in COLUMN_ALIASES.items():
        for c, n in norm.items():
            if n in aliases or n.replace("_", "") in [a.replace("_", "") for a in aliases]:
                out[field] = c
                break
    return out


@dataclass
class ValidationResult:
    ok: bool
    errors: list
    warnings: list
    data: pd.DataFrame | None


def validate_new_data(df: pd.DataFrame, mapping: dict) -> ValidationResult:
    """Validate an uploaded CSV of transactions or weekly demand and standardise it to
    columns week, sku, demand, avg_price (avg_price optional)."""
    errors, warnings = [], []
    missing = [f for f in REQUIRED_FIELDS if not mapping.get(f) or mapping[f] not in df.columns]
    if missing:
        return ValidationResult(False, [f"Required field(s) not mapped: {', '.join(missing)}"], [], None)
    out = pd.DataFrame({
        "date": pd.to_datetime(df[mapping["date"]], errors="coerce", dayfirst=False),
        "sku": df[mapping["sku"]].astype(str).str.upper().str.strip(),
        "quantity": pd.to_numeric(df[mapping["quantity"]], errors="coerce"),
    })
    if mapping.get("price") in df.columns:
        out["price"] = pd.to_numeric(df[mapping["price"]], errors="coerce")
    bad_date = int(out["date"].isna().sum())
    bad_qty = int(out["quantity"].isna().sum())
    if bad_date:
        warnings.append(f"{bad_date} row(s) with an unreadable date were dropped.")
    if bad_qty:
        warnings.append(f"{bad_qty} row(s) with a non-numeric quantity were dropped.")
    out = out.dropna(subset=["date", "quantity"])
    neg = int((out["quantity"] < 0).sum())
    if neg:
        warnings.append(f"{neg} negative-quantity row(s) (returns) were excluded from demand.")
        out = out[out["quantity"] >= 0]
    dups = int(out.duplicated().sum())
    if dups:
        warnings.append(f"{dups} exact duplicate row(s) were removed.")
        out = out.drop_duplicates()
    if out.empty:
        errors.append("No valid rows remain after validation.")
        return ValidationResult(False, errors, warnings, None)
    out["week"] = out["date"].dt.to_period("W-SUN").dt.start_time
    agg = {"demand": ("quantity", "sum")}
    if "price" in out:
        out["rev"] = out["quantity"] * out["price"]
        agg["rev"] = ("rev", "sum")
    wk = out.groupby(["sku", "week"]).agg(**agg).reset_index()
    if "rev" in wk:
        wk["avg_price"] = wk["rev"] / wk["demand"].replace(0, np.nan)
        wk = wk.drop(columns="rev")
    return ValidationResult(True, errors, warnings, wk)


def append_weekly(panel: pd.DataFrame, new_weekly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Append or reconcile new weekly observations (new values replace overlapping weeks)
    for SKUs already tracked. Recomputes derived and external features."""
    known = set(panel["sku"])
    nw = new_weekly[new_weekly["sku"].isin(known)].copy()
    info = {"rows_in": int(len(new_weekly)), "rows_used": int(len(nw)),
            "unknown_skus": sorted(set(new_weekly["sku"]) - known)[:10]}
    if nw.empty:
        info.update(new_weeks=0, replaced=0)
        return panel, info
    key = ["sku", "week"]
    overlap = panel.merge(nw[key], on=key)
    info["replaced"] = int(len(overlap))
    base = panel.set_index(key)
    nw = nw.set_index(key)
    base = base.drop(index=nw.index.intersection(base.index))
    desc = panel.groupby("sku")["description"].first()
    nw["description"] = nw.index.get_level_values("sku").map(desc)
    out = pd.concat([base, nw]).reset_index().sort_values(key)
    # fill gaps for SKUs without sales in a new week
    weeks = pd.date_range(out["week"].min(), out["week"].max(), freq="W-MON")
    full = pd.MultiIndex.from_product([sorted(known), weeks], names=key)
    out = out.set_index(key).reindex(full).reset_index()
    out["demand"] = out["demand"].fillna(0.0)
    out["description"] = out["sku"].map(desc)
    out["avg_price"] = out.groupby("sku")["avg_price"].transform(lambda s: s.ffill().bfill())
    med = out.groupby("sku")["avg_price"].transform("median")
    out["price_cut"] = (out["avg_price"] < 0.9 * med).astype(int)
    out["holidays"] = uk_holidays_per_week(pd.DatetimeIndex(out["week"])).values
    info["new_weeks"] = int(out["week"].nunique() - panel["week"].nunique())
    return out[panel.columns], info


def panel_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()
