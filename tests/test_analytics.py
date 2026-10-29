"""Tests for src.analytics — RFM and product analysis on synthetic sales."""
import numpy as np
import pandas as pd

from src.analytics import product_analysis, rfm_analysis

PRODUCT_LINES = [
    "Health and beauty",
    "Electronic accessories",
    "Home and lifestyle",
    "Sports and travel",
    "Food and beverages",
    "Fashion accessories",
]

VALID_SEGMENTS = {
    "Champions", "Loyal", "At Risk", "Lost", "New Customers", "Potential Loyalists",
}


def _make_mock_sales(n: int = 200) -> pd.DataFrame:
    """Build n synthetic sales rows spread over ~90 days."""
    rng = np.random.default_rng(42)
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 90, size=n), unit="D"
    )
    totals = rng.uniform(10.0, 200.0, size=n).round(2)
    return pd.DataFrame({
        "Invoice_ID": [f"INV-{i:06d}" for i in range(n)],
        "Date": dates,
        "Product_Line": rng.choice(PRODUCT_LINES, size=n),
        "Quantity": rng.integers(1, 10, size=n),
        "Total": totals,
        "Gross_Income": (totals * 0.0476).round(2),
    })


def test_rfm_scores_range():
    """R, F and M scores are all integers in [1, 5]."""
    rfm = rfm_analysis(_make_mock_sales(200))
    for column in ("R_Score", "F_Score", "M_Score"):
        assert rfm[column].between(1, 5).all(), f"{column} out of range"


def test_rfm_segments_valid():
    """Every RFM segment is one of the six rule-based segments."""
    rfm = rfm_analysis(_make_mock_sales(200))
    assert set(rfm["RFM_Segment"]).issubset(VALID_SEGMENTS)


def test_clv_positive():
    """All CLV values are strictly positive with a valid tier."""
    rfm = rfm_analysis(_make_mock_sales(200))
    assert (rfm["CLV"] > 0).all()
    assert set(rfm["CLV_Tier"]).issubset({"Low", "Medium", "High"})


def test_product_analysis_keys():
    """product_analysis returns best sellers, profitability and ABC output."""
    result = product_analysis(_make_mock_sales(200))
    for key in ("best_sellers", "slow_movers", "profitability", "seasonal", "abc", "charts"):
        assert key in result, f"missing key: {key}"
    assert not result["best_sellers"].empty
    assert {"Revenue", "Profit", "Margin_Pct"}.issubset(result["profitability"].columns)
    assert "ABC_Class" in result["abc"].columns
    assert set(result["abc"]["ABC_Class"]).issubset({"A", "B", "C"})
    assert result["charts"], "expected at least one plotly chart"
