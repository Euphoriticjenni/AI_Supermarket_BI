"""Shared data access for the NexusBI dashboard.

Single home for cached loaders, the shared sidebar filter, and cached
wrappers around the expensive analytics / forecasting engines — so no page
re-loads the 471K-row table or recomputes RFM/trends/Prophet on every widget
change.

Pure functions (no Streamlit calls) are grouped at the top and covered by
``tests/test_dashboard_helpers.py``:
``filter_sales_df``, ``previous_window``, ``period_kpis``,
``daily_revenue``, ``revenue_by``, ``template_match``, ``severity_from_action``.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Mapping, Optional, Tuple

import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_HERE, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import theme  # noqa: E402  (dashboard/theme.py — same directory)

from src import config  # noqa: E402
from src.ai_assistant import BusinessAssistant  # noqa: E402
from src.analytics import rfm_analysis, trend_analysis  # noqa: E402
from src.forecasting import evaluate_backtest, fit_forecast  # noqa: E402
from src.inventory import (  # noqa: E402
    abc_analysis,
    calculate_safety_stock,
    generate_reorder_report,
)


# ===========================================================================
# Pure helpers (no Streamlit, no caching — unit-tested)
# ===========================================================================
def filter_sales_df(
    df: pd.DataFrame,
    start: date,
    end: date,
    branches: Optional[Iterable[str]] = None,
    cities: Optional[Iterable[str]] = None,
    product_lines: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Filter the sales table by an inclusive date range and dimensions.

    ``None`` / empty / ``["All"]`` selections mean "no restriction".
    """
    mask = (df["Date"] >= pd.Timestamp(start)) & (
        df["Date"] < pd.Timestamp(end) + pd.Timedelta(days=1)
    )
    if branches and "All" not in branches:
        mask &= df["Branch"].isin(branches)
    if cities and "All" not in cities:
        mask &= df["City"].isin(cities)
    if product_lines and "All" not in product_lines:
        mask &= df["Product_Line"].isin(product_lines)
    return df.loc[mask]


def previous_window(start: date, end: date) -> Tuple[date, date]:
    """The window of equal length immediately before ``start`` (inclusive)."""
    days = (end - start).days + 1
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)
    return prev_start, prev_end


def period_kpis(df: pd.DataFrame) -> dict:
    """Headline figures for one filtered slice (real aggregations only)."""
    orders = int(df["Invoice_ID"].nunique())
    revenue = float(df["Total"].sum()) if len(df) else 0.0
    return {
        "revenue": revenue,
        "gross_income": float(df["Gross_Income"].sum()) if len(df) else 0.0,
        "orders": orders,
        "aov": (revenue / orders) if orders else None,
    }


def daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue + order count per calendar day (sorted) — pure."""
    days = df["Date"].dt.normalize().rename("Day")
    return (
        df.assign(Day=days)
        .groupby("Day", as_index=False)
        .agg(Revenue=("Total", "sum"), Orders=("Invoice_ID", "nunique"))
        .sort_values("Day")
    )


def revenue_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Total revenue per dimension value, highest first — pure."""
    return (
        df.groupby(column, observed=True)["Total"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def template_match(templates: Mapping[str, str], question: str) -> Optional[str]:
    """Mirror ``BusinessAssistant._match_template``'s scan (lowercase, ordered).

    Returns the first matching pattern, or ``None`` when the question maps to
    no template — the page then shows guidance instead of the overview answer.
    """
    lowered = question.lower()
    for pattern in templates:
        if re.search(pattern, lowered):
            return pattern
    return None


#: Plain-text severity for each emoji Action emitted by the reorder report —
#: the UI never relies on emoji alone to communicate urgency.
SEVERITY_BY_ACTION = {
    "🔴 URGENT REORDER": "Urgent",
    "🟡 REORDER SOON": "Reorder soon",
    "🟢 ADEQUATE": "OK",
    "🔵 OVERSTOCK": "Overstocked",
}


def severity_from_action(action: str) -> str:
    """Plain-text severity for a reorder-report Action (pure, tested)."""
    return SEVERITY_BY_ACTION.get(str(action), "Unknown")


# ===========================================================================
# Cached loaders (friendly errors instead of tracebacks)
# ===========================================================================
@st.cache_data(show_spinner="Loading sales data …")
def _load_sales_frame() -> Optional[pd.DataFrame]:
    parquet_path = config.PROCESSED_DATA_PATH / "sales_features.parquet"
    csv_path = config.AUGMENTED_DATA_PATH / "supermarket_sales.csv"
    try:
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["Date"])
        else:
            return None
    except Exception:  # unreadable/corrupt file → same friendly path
        return None
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Categorical dtype makes value_counts() report zero-count categories
    # after filtering; plain strings keep the charts honest.
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    return df


def load_sales() -> pd.DataFrame:
    """Sales table, or a friendly stop with pipeline instructions."""
    df = _load_sales_frame()
    if df is None:
        theme.empty_state(
            "Sales data not found",
            "Run the pipeline first: `make generate-data && make preprocess && "
            f"make features` (expected `{config.PROCESSED_DATA_PATH / 'sales_features.parquet'}`).",
        )
        st.stop()
    return df


@st.cache_data(show_spinner="Loading product catalogue …")
def load_products() -> Optional[pd.DataFrame]:
    if not config.PRODUCTS_DATA_FILE.exists():
        return None
    return pd.read_csv(config.PRODUCTS_DATA_FILE)


def products_or_stop() -> pd.DataFrame:
    products = load_products()
    if products is None:
        theme.empty_state(
            "Product catalogue not found",
            f"Run `make generate-data` first (expected `{config.PRODUCTS_DATA_FILE}`).",
        )
        st.stop()
    return products


@st.cache_data(show_spinner="Loading customer base …")
def load_customers() -> Optional[pd.DataFrame]:
    """Customer base records (real catalogue rows — distinct from the RFM proxy)."""
    if config.CUSTOMERS_DATA_FILE.exists():
        return pd.read_csv(config.CUSTOMERS_DATA_FILE)
    return None


# ===========================================================================
# Shared sidebar filters
# ===========================================================================
@dataclass(frozen=True)
class FilterSelection:
    start: date
    end: date
    branches: Tuple[str, ...]
    cities: Tuple[str, ...]
    product_lines: Tuple[str, ...]


def _multiselect(label: str, column: str, df: pd.DataFrame) -> Tuple[str, ...]:
    options = ["All"] + sorted(df[column].dropna().astype(str).unique())
    picked = st.sidebar.multiselect(
        label,
        options,
        default=["All"],
        help='"All" includes every value; an empty box also means no restriction.',
    )
    return tuple(picked)


def render_filters(df: pd.DataFrame) -> FilterSelection:
    """Sidebar filter widgets — dates, branch, city, product line."""
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    picked = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Transactions outside this range are excluded from every chart.",
    )
    if isinstance(picked, (tuple, list)):
        start = picked[0] if len(picked) >= 1 else min_date
        end = picked[1] if len(picked) >= 2 else (picked[0] if len(picked) == 1 else max_date)
    elif picked is not None:
        start = end = picked
    else:
        start, end = min_date, max_date

    return FilterSelection(
        start=start,
        end=end,
        branches=_multiselect("Branch", "Branch", df),
        cities=_multiselect("City", "City", df),
        product_lines=_multiselect("Product line", "Product_Line", df),
    )


def filtered(df: pd.DataFrame, sel: FilterSelection) -> pd.DataFrame:
    """Current selection (boolean indexing already returns a new frame)."""
    return filter_sales_df(df, sel.start, sel.end, sel.branches, sel.cities, sel.product_lines)


def filtered_with_previous(
    df: pd.DataFrame, sel: FilterSelection
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Current slice plus the preceding equal-length window (or None)."""
    current = filtered(df, sel)
    prev_start, prev_end = previous_window(sel.start, sel.end)
    has_room = (df["Date"].min().date() <= prev_end) and prev_start < sel.start
    previous = (
        filter_sales_df(df, prev_start, prev_end, sel.branches, sel.cities, sel.product_lines)
        if has_room
        else None
    )
    return current, previous


# ===========================================================================
# Cached analytics / forecasting wrappers (never recomputed on widget changes)
# ===========================================================================
@st.cache_data(show_spinner="Computing RFM segments (rule-based) …")
def compute_rfm(sales: pd.DataFrame) -> pd.DataFrame:
    """Cached wrapper around ``src.analytics.rfm_analysis`` (rule-based)."""
    return rfm_analysis(sales[["Invoice_ID", "Date", "Total"]])


@st.cache_data(show_spinner="Computing trends & projection …")
def compute_trends(sales: pd.DataFrame) -> dict:
    """Cached wrapper around ``src.analytics.trend_analysis``."""
    return trend_analysis(sales[["Date", "Total"]])


@st.cache_data(show_spinner="Computing safety stock …")
def compute_safety_stock(
    products: pd.DataFrame, sales: pd.DataFrame, service_level: float, lead_time: int
) -> pd.DataFrame:
    safety = calculate_safety_stock(
        products, sales[["Date", "Product_Line", "Quantity"]],
        service_level=service_level, lead_time=lead_time,
    )
    demand_cols = ["Avg_Daily_Demand", "Demand_Std", "Safety_Stock", "Reorder_Point"]
    if safety[demand_cols].isna().any().any():
        safety[demand_cols] = safety[demand_cols].fillna(0.0)
    return safety


@st.cache_data(show_spinner="Building reorder report …")
def compute_reorder_report(safety: pd.DataFrame) -> pd.DataFrame:
    return generate_reorder_report(safety)


@st.cache_data(show_spinner="Running ABC analysis …")
def compute_abc(sales: pd.DataFrame) -> dict:
    return abc_analysis(sales[["Product_Line", "Total"]])


@st.cache_data(show_spinner="Fitting Prophet forecast …")
def compute_forecast(
    daily: pd.DataFrame, periods: int, changepoint_prior_scale: float
) -> dict:
    """Cached Prophet fit + forecast (daily = ds/y frame, see src.forecasting)."""
    return fit_forecast(
        daily, periods=periods, changepoint_prior_scale=changepoint_prior_scale
    )


@st.cache_data(show_spinner="Running backtest …")
def compute_backtest(daily: pd.DataFrame, holdout_days: int) -> dict:
    """Cached holdout evaluation of the Prophet engine."""
    return evaluate_backtest(daily, holdout_days=holdout_days)


@st.cache_resource(show_spinner="Starting business assistant …")
def get_assistant() -> BusinessAssistant:
    """Singleton assistant — rule-based regex→SQL over the SQLite store DB."""
    return BusinessAssistant()


@st.cache_data(show_spinner=False)
def executive_summary() -> str:
    return get_assistant().get_executive_summary()


@st.cache_data(show_spinner=False)
def assistant_insights() -> list:
    return get_assistant().get_insights()
