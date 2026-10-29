"""Tests for src.inventory — safety stock and reorder reporting."""
import numpy as np
import pandas as pd

from src.inventory import calculate_safety_stock, generate_reorder_report

URGENT = "🔴 URGENT REORDER"
REORDER = "🟡 REORDER SOON"
ADEQUATE = "🟢 ADEQUATE"
OVERSTOCK = "🔵 OVERSTOCK"


def _make_mock_inventory():
    """4 products (one with Stock=0) plus 30 days of matching sales."""
    products = pd.DataFrame({
        "Product_ID": ["PROD-0001", "PROD-0002", "PROD-0003", "PROD-0004"],
        "Product_Name": ["Alpha", "Beta", "Gamma", "Delta"],
        "Category": [
            "Food and beverages",
            "Health and beauty",
            "Home and lifestyle",
            "Sports and travel",
        ],
        "Cost": [5.0, 8.0, 12.0, 20.0],
        "Selling_Price": [8.0, 12.0, 18.0, 30.0],
        "Stock": [50, 0, 200, 5],
        "Reorder_Level": [20, 20, 20, 20],
    })

    rng = np.random.default_rng(7)
    rows = []
    for day in range(30):
        date = pd.Timestamp("2023-06-01") + pd.Timedelta(days=day)
        for line in products["Category"]:
            for _ in range(int(rng.integers(1, 4))):  # 1-3 transactions/day
                rows.append({
                    "Date": date,
                    "Product_Line": line,
                    "Quantity": int(rng.integers(1, 6)),
                    "Total": 100.0,
                })
    sales = pd.DataFrame(rows)
    return products, sales


def test_safety_stock_positive():
    """Safety stock and reorder points are non-negative with all columns."""
    products, sales = _make_mock_inventory()
    ss = calculate_safety_stock(products, sales, service_level=0.95, lead_time=7)

    for column in ("Product_ID", "Current_Stock", "Avg_Daily_Demand",
                   "Demand_Std", "Safety_Stock", "Reorder_Point"):
        assert column in ss.columns, f"missing column: {column}"
    assert (ss["Safety_Stock"] >= 0).all()
    assert (ss["Reorder_Point"] >= 0).all()
    assert (ss["Reorder_Point"] >= ss["Safety_Stock"]).all()


def test_reorder_actions():
    """The zero-stock product is flagged URGENT REORDER."""
    products, sales = _make_mock_inventory()
    ss = calculate_safety_stock(products, sales)
    report = generate_reorder_report(ss)

    beta = report.loc[report["Product_ID"] == "PROD-0002"].iloc[0]
    assert beta["Current_Stock"] == 0
    assert beta["Action"] == URGENT


def test_reorder_report_sorted():
    """URGENT rows sort before ADEQUATE/OVERSTOCK rows."""
    products, sales = _make_mock_inventory()
    ss = calculate_safety_stock(products, sales)
    report = generate_reorder_report(ss)

    order = [URGENT, REORDER, ADEQUATE, OVERSTOCK]
    present = [a for a in order if a in set(report["Action"])]
    # Actions appear in report exactly in the canonical priority order.
    seen_in_report = report["Action"].drop_duplicates().tolist()
    assert seen_in_report == [a for a in seen_in_report if a in present]
    assert seen_in_report == present  # full priority ordering preserved

    if URGENT in seen_in_report and ADEQUATE in seen_in_report:
        first_urgent = report.index[report["Action"] == URGENT][0]
        first_adequate = report.index[report["Action"] == ADEQUATE][0]
        assert first_urgent < first_adequate
