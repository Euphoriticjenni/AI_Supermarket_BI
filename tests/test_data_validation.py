"""Tests for src.data_validation — pandera schema checks on tiny frames."""
import pandas as pd

from src.data_validation import validate_dataset


def _valid_sales(n: int = 10) -> pd.DataFrame:
    """Build n minimal, fully valid sales rows."""
    return pd.DataFrame({
        "Invoice_ID": [f"INV-{i:06d}" for i in range(n)],
        "Date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "Branch": ["A"] * n,
        "City": ["Yangon"] * n,
        "Unit_Price": [25.0] * n,
        "Quantity": [3] * n,
        "Total": [78.75] * n,
        "Rating": [7.0] * n,
        "COGS": [75.0] * n,
    })


def test_valid_sales_data():
    """10 valid rows pass the sales schema."""
    report = validate_dataset(_valid_sales(10), schema_type="sales")
    assert report["passed"] is True
    assert report["failures"] == 0


def test_invalid_negative_price():
    """A negative Unit_Price fails the gt(0) check."""
    df = _valid_sales(10)
    df.loc[3, "Unit_Price"] = -5.0
    report = validate_dataset(df, schema_type="sales")
    assert report["passed"] is False
    assert report["failures"] >= 1


def test_invalid_rating_range():
    """Rating 15 is outside [1, 10] and must fail."""
    df = _valid_sales(10)
    df.loc[5, "Rating"] = 15.0
    report = validate_dataset(df, schema_type="sales")
    assert report["passed"] is False
    assert report["failures"] >= 1


def test_missing_required_columns():
    """Dropping the Branch column fails schema validation."""
    df = _valid_sales(10).drop(columns=["Branch"])
    report = validate_dataset(df, schema_type="sales")
    assert report["passed"] is False
