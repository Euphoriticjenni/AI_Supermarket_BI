"""Integration tests for fixes applied to handle new schema and dashboard reliability."""
import pandas as pd
import pytest

# Test feature engineering column resolution
def test_feature_engineering_new_schema():
    import pandas as pd
    from src.feature_engineering import add_temporal_features, add_financial_features, run_feature_engineering

    # Minimal new-schema frame
    df = pd.DataFrame({
        "transaction_id": ["TXN-00000001", "TXN-00000002"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "time": ["08:00:00", "14:30:00"],
        "category": ["Fresh Produce", "Beverages"],
        "quantity": [2, 1],
        "unit_price": [10.0, 5.0],
        "unit_cost": [6.0, 3.0],
        "total_amount": [20.0, 5.0],
        "profit": [8.0, 2.0],
        "payment_method": ["Cash", "Cash"],
        "branch": ["Downtown Central", "Downtown Central"],
    })
    # Should not raise
    out = add_temporal_features(df.copy())
    assert "Day_of_Week" in out.columns
    assert "Purchase_Hour" in out.columns

    out2 = add_financial_features(out.copy())
    assert "Profit" in out2.columns

    # Full pipeline on synthetic 60-day history
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    big = pd.DataFrame({
        "transaction_id": [f"TXN-{i:08d}" for i in range(60)],
        "date": dates,
        "time": ["10:00:00"]*60,
        "category": ["Fresh Produce"]*60,
        "quantity": [2]*60,
        "unit_price": [10.0]*60,
        "unit_cost": [6.0]*60,
        "total_amount": [20.0]*60,
        "profit": [8.0]*60,
        "payment_method": ["Cash"]*60,
        "branch": ["Downtown Central"]*60,
    })
    result = run_feature_engineering(big, save=False)
    assert "Rolling_7d_Sales" in result.columns


def test_analytics_new_schema():
    from src.analytics import rfm_analysis, product_analysis, trend_analysis
    import pandas as pd
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "transaction_id": [f"TXN-{i:08d}" for i in range(100)],
        "date": dates,
        "category": ["Beverages" if i%2==0 else "Fresh Produce" for i in range(100)],
        "quantity": [1]*100,
        "total_amount": [20.0]*100,
        "profit": [5.0]*100,
    })
    rfm = rfm_analysis(df)
    assert not rfm.empty
    assert "RFM_Segment" in rfm.columns

    prod = product_analysis(df)
    assert "Product_Line" in prod["best_sellers"].columns
    assert len(prod["best_sellers"]) <= 10

    trend = trend_analysis(df)
    assert "projection" in trend
    assert "projected_total" in trend["projection"]


def test_inventory_new_schema():
    from src.inventory import analyze_stock, calculate_safety_stock, abc_analysis
    import pandas as pd

    # Sales with new schema
    sales = pd.DataFrame({
        "transaction_id": [f"TXN-{i:08d}" for i in range(100)],
        "date": pd.to_datetime(["2024-01-01"]*100),
        "category": ["Fresh Produce"]*50 + ["Beverages"]*50,
        "quantity": [2]*100,
        "total_amount": [20.0]*100,
    })
    products = pd.DataFrame({
        "product_id": [f"PROD-{i:05d}" for i in range(10)],
        "name": [f"Prod {i}" for i in range(10)],
        "category": ["Fresh Produce"]*5 + ["Beverages"]*5,
        "unit_cost": [5.0]*10,
        "unit_price": [10.0]*10,
        "stock": [100]*10,
        "reorder_level": [20]*10,
        "supplier": ["Supp"]*10,
    })
    stock = analyze_stock(products, sales)
    assert "summary" in stock
    assert stock["summary"]["total_products"] == 10

    safety = calculate_safety_stock(products, sales)
    assert "Safety_Stock" in safety.columns

    abc = abc_analysis(sales)
    assert "abc_df" in abc
    assert not abc["abc_df"].empty


def test_theme_load_data_integration():
    from dashboard.theme import load_data
    df = load_data()
    assert not df.empty
    assert "total_amount" in df.columns
    assert "date" in df.columns
    # Check that all dashboard pages' required columns exist
    for col in ["transaction_id", "branch", "category", "payment_method", "quantity", "profit", "customer_id"]:
        assert col in df.columns, f"missing {col}"


def test_forecasting_with_new_schema():
    from src.forecasting import DemandForecaster
    import pandas as pd
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "total_amount": [100.0 + i*0.5 for i in range(60)],
    })
    model = DemandForecaster(periods=10).fit(df, date_col="date", value_col="total_amount")
    pred = model.predict(10)
    assert len(pred) == 10
    assert "yhat" in pred.columns
    summary = model.get_forecast_summary()
    assert "next_7d" in summary
    assert summary["backend"] in ("prophet", "ema")


def test_page_required_columns_guard():
    # Simulate missing column scenario: pages should detect missing columns
    from dashboard.theme import load_data
    df = load_data()
    required = ['transaction_id', 'date', 'branch', 'category', 'payment_method', 'quantity', 'total_amount', 'customer_id', 'datetime', 'hour_of_day']
    missing = [c for c in required if c not in df.columns]
    assert missing == [], f"Theme should provide all required columns, missing {missing}"