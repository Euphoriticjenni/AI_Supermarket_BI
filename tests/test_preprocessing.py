"""Tests for src.preprocessing — cleaning functions on tiny synthetic frames."""
import numpy as np
import pandas as pd

from src.preprocessing import (
    fix_data_types,
    handle_missing_values,
    remove_duplicates,
    remove_outliers,
    standardize_categories,
)


def test_remove_duplicates():
    """10 rows containing 3 exact duplicates -> 7 unique rows remain."""
    base = pd.DataFrame({
        "Invoice_ID": [f"INV-{i:03d}" for i in range(7)],
        "Total": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
    })
    df = pd.concat([base, base.iloc[[0, 3, 6]]], ignore_index=True)  # 10 rows
    assert len(df) == 10

    out = remove_duplicates(df)
    assert len(out) == 7
    assert out["Invoice_ID"].is_unique


def test_handle_missing_values_numeric():
    """3 NaNs in a numeric column are filled with the median of the rest."""
    df = pd.DataFrame({
        "Rating": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                   np.nan, np.nan, np.nan],
    })
    out = handle_missing_values(df)
    assert not out["Rating"].isna().any()
    # Median of the 9 non-null values [1..9] is 5.0 -> 3 imputed cells plus
    # the one original 5.0 make four fives in total.
    assert (out["Rating"] == 5.0).sum() == 4


def test_handle_missing_values_categorical():
    """2 NaNs in a categorical column are filled with the mode."""
    df = pd.DataFrame({
        "City": ["Yangon", "Yangon", "Mandalay", "Yangon", np.nan, np.nan],
    })
    out = handle_missing_values(df)
    assert not out["City"].isna().any()
    assert (out["City"] == "Yangon").sum() == 5  # 3 original + 2 imputed


def test_fix_data_types():
    """String dates become datetime64; category columns are encoded."""
    df = pd.DataFrame({
        "Date": ["2022-01-01", "2022-01-02", "2022-01-03"],
        "Branch": ["A", "B", "C"],
        "Total": ["10.5", "20.0", "30.25"],
        "Quantity": ["1", "2", "3"],
    })
    out = fix_data_types(df)
    assert pd.api.types.is_datetime64_any_dtype(out["Date"])
    assert pd.api.types.is_float_dtype(out["Total"])
    assert pd.api.types.is_integer_dtype(out["Quantity"])
    assert isinstance(out["Branch"].dtype, pd.CategoricalDtype)


def test_remove_outliers():
    """5 extreme values among 100 normal ones are quarantined separately.

    A few normal samples may legitimately fall outside the IQR bounds
    (~4% of a normal distribution), so the assertion is that every extreme
    is quarantined and none survives in the clean frame.
    """
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=100.0, scale=5.0, size=100)
    df = pd.DataFrame({"Total": np.concatenate([normal, [10_000.0] * 5])})

    clean, anomalies = remove_outliers(df, columns=["Total"])
    assert len(clean) + len(anomalies) == len(df)
    assert (anomalies["Total"] > 1000).all()      # all 5 extremes quarantined
    assert clean["Total"].max() < 1000            # none survived in clean
    assert clean["Total"].min() < 200             # normal data kept


def test_standardize_categories():
    """Whitespace is stripped and messy category text becomes canonical."""
    df = pd.DataFrame({
        "Product_Line": ["  health and BEAUTY ", "food AND beverages"],
        "City": [" yangon ", "MANDALAY"],
        "Payment": ["credit card", "CASH"],
        "Gender": [" male ", "FEMALE"],
        "Customer_Type": [" member ", "normal"],
    })
    out = standardize_categories(df)
    assert out["Product_Line"].tolist() == ["Health and beauty", "Food and beverages"]
    assert out["City"].tolist() == ["Yangon", "Mandalay"]
    assert out["Payment"].tolist() == ["Credit card", "Cash"]
    assert out["Gender"].tolist() == ["Male", "Female"]
    assert out["Customer_Type"].tolist() == ["Member", "Normal"]
