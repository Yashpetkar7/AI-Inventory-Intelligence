"""Shared AI inventory-intelligence engine (Lab 2, AI in Operations).
Used by both the Streamlit dashboard and the submission notebook, so every number agrees."""
from . import data, forecasting, inventory, simulation, optimization  # noqa: F401
__version__ = "1.0.0"
