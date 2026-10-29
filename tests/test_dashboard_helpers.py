"""Tests for the shared dashboard helpers.

Covers the pure calculation / filtering logic introduced with the dashboard
refactor: ``dashboard/data.py`` (filters, comparison windows, KPI math,
aggregations, template matching, severity mapping) and the formatting
helpers in ``dashboard/ui.py``.
"""
import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    """Load a dashboard module by file path (dashboard/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ui = _load("dash_ui_under_test", "dashboard/theme.py")
dash = _load("dash_data_under_test", "dashboard/data.py")


@pytest.fixture
def sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2022-12-30", "2023-01-05", "2023-01-06", "2023-02-10"]
            ),
            "Branch": ["A", "A", "B", "B"],
            "City": ["Yangon", "Yangon", "Mandalay", "Mandalay"],
            "Product_Line": ["Food", "Fashion", "Food", "Home"],
            "Total": [100.0, 50.0, 200.0, 80.0],
            "Gross_Income": [5.0, 2.5, 10.0, 4.0],
            "Invoice_ID": ["I1", "I2", "I3", "I4"],
        }
    )


# ---------------------------------------------------------------------------
# filter_sales_df
# ---------------------------------------------------------------------------
def test_filter_by_date_range(sales_df):
    out = dash.filter_sales_df(sales_df, date(2023, 1, 1), date(2023, 1, 31))
    assert list(out["Invoice_ID"]) == ["I2", "I3"]


def test_filter_is_end_inclusive(sales_df):
    out = dash.filter_sales_df(sales_df, date(2023, 2, 10), date(2023, 2, 10))
    assert list(out["Invoice_ID"]) == ["I4"]


def test_filter_by_branch_and_city(sales_df):
    out = dash.filter_sales_df(
        sales_df, date(2022, 12, 1), date(2023, 3, 1), branches=["B"], cities=["Mandalay"]
    )
    assert list(out["Invoice_ID"]) == ["I3", "I4"]


def test_filter_by_product_line(sales_df):
    out = dash.filter_sales_df(
        sales_df, date(2022, 12, 1), date(2023, 3, 1), product_lines=["Food"]
    )
    assert set(out["Invoice_ID"]) == {"I1", "I3"}


def test_filter_all_means_no_restriction(sales_df):
    out = dash.filter_sales_df(
        sales_df, date(2022, 12, 1), date(2023, 3, 1),
        branches=["All"], cities=["All"], product_lines=["All"],
    )
    assert len(out) == 4


def test_filter_empty_selection_means_no_restriction(sales_df):
    out = dash.filter_sales_df(
        sales_df, date(2022, 12, 1), date(2023, 3, 1), branches=[], cities=[]
    )
    assert len(out) == 4


# ---------------------------------------------------------------------------
# previous_window + KPI math
# ---------------------------------------------------------------------------
def test_previous_window_is_equal_length_and_adjacent():
    assert dash.previous_window(date(2023, 1, 5), date(2023, 1, 6)) == (
        date(2023, 1, 3),
        date(2023, 1, 4),
    )


def test_previous_window_single_day():
    assert dash.previous_window(date(2023, 3, 1), date(2023, 3, 1)) == (
        date(2023, 2, 28),
        date(2023, 2, 28),
    )


def test_previous_window_captures_prior_data(sales_df):
    prev_start, prev_end = dash.previous_window(date(2023, 1, 5), date(2023, 2, 10))
    prev = dash.filter_sales_df(sales_df, prev_start, prev_end)
    assert list(prev["Invoice_ID"]) == ["I1"]


def test_period_kpis(sales_df):
    kpis = dash.period_kpis(sales_df)
    assert kpis["revenue"] == pytest.approx(430.0)
    assert kpis["gross_income"] == pytest.approx(21.5)
    assert kpis["orders"] == 4
    assert kpis["aov"] == pytest.approx(107.5)


def test_period_kpis_empty_frame(sales_df):
    empty = sales_df.iloc[0:0]
    kpis = dash.period_kpis(empty)
    assert kpis["revenue"] == 0.0
    assert kpis["orders"] == 0
    assert kpis["aov"] is None


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
def test_daily_revenue_groups_and_sorts(sales_df):
    daily = dash.daily_revenue(sales_df)
    assert list(daily["Day"].dt.strftime("%Y-%m-%d")) == [
        "2022-12-30", "2023-01-05", "2023-01-06", "2023-02-10",
    ]
    assert daily.loc[daily["Day"] == "2023-01-06", "Revenue"].iloc[0] == pytest.approx(200.0)


def test_revenue_by_sorted_descending(sales_df):
    by_line = dash.revenue_by(sales_df, "Product_Line")
    assert list(by_line["Product_Line"]) == ["Food", "Home", "Fashion"]
    assert by_line["Total"].iloc[0] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Template matching (mirrors BusinessAssistant._match_template)
# ---------------------------------------------------------------------------
TEMPLATES = {
    r"top products|best selling": "SELECT 1",
    r"stock|restock": "SELECT 2",
    r"overview|summary|kpi": "SELECT 3",
}


def test_template_match_first_match_wins():
    assert dash.template_match(TEMPLATES, "Show me the top products") == r"top products|best selling"


def test_template_match_is_case_insensitive():
    assert dash.template_match(TEMPLATES, "TOP PRODUCTS please") == r"top products|best selling"


def test_template_match_no_match_returns_none():
    assert dash.template_match(TEMPLATES, "What is the meaning of life?") is None


# ---------------------------------------------------------------------------
# Severity mapping (reorder report)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("action", "severity"),
    [
        ("🔴 URGENT REORDER", "Urgent"),
        ("🟡 REORDER SOON", "Reorder soon"),
        ("🟢 ADEQUATE", "OK"),
        ("🔵 OVERSTOCK", "Overstocked"),
        ("something unexpected", "Unknown"),
    ],
)
def test_severity_from_action(action, severity):
    assert dash.severity_from_action(action) == severity


# ---------------------------------------------------------------------------
# ui formatting helpers
# ---------------------------------------------------------------------------
def test_currency_compact_scales():
    assert ui.currency(82_664_280) == "$82.7M"
    assert ui.currency(431_500) == "$431.5K"
    assert ui.currency(208.75) == "$208.75"
    assert ui.currency(0) == "$0.00"


def test_currency_handles_missing():
    assert ui.currency(None) == "—"


def test_currency_full_precision():
    assert ui.currency(1_234_567.891, compact=False) == "$1,234,567.89"


def test_fmt_int():
    assert ui.fmt_int(471_588) == "471,588"
    assert ui.fmt_int(None) == "—"


def test_pct_change_basic():
    assert ui.pct_change(150.0, 100.0) == pytest.approx(50.0)
    assert ui.pct_change(50.0, 200.0) == pytest.approx(-75.0)


def test_pct_change_omits_when_uncomputable():
    assert ui.pct_change(100.0, 0.0) is None
    assert ui.pct_change(None, 100.0) is None
    assert ui.pct_change(100.0, None) is None
