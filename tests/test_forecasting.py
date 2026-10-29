"""Tests for src.forecasting — pure helpers of the Prophet engine.

Only deterministic, Prophet-free logic is unit-tested here (aggregation,
splits, metrics, capability probe) so the suite stays fast; the Prophet fits
themselves are exercised by the dashboard's cached wrappers.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fc = _load("src_forecasting_under_test", "src/forecasting.py")


@pytest.fixture
def daily() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=120, freq="D")
    return pd.DataFrame({
        "ds": idx,
        "y": np.linspace(100.0, 220.0, len(idx)) + np.sin(np.arange(len(idx))) * 10,
    })


def test_is_available_probe():
    assert fc.is_available() is True or fc.availability_error() is not None


def test_prepare_daily_aggregates_and_sorts():
    sales = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-02 10:00", "2023-01-02 18:00", "2023-01-01 09:00"]),
        "Total": [100.0, 50.0, 200.0],
    })
    daily = fc.prepare_daily(sales)
    assert list(daily["ds"]) == list(pd.to_datetime(["2023-01-01", "2023-01-02"]))
    assert daily["y"].tolist() == [200.0, 150.0]


def test_prepare_daily_handles_missing_dates():
    sales = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01", "2023-01-03"]),
        "Total": [10.0, 30.0],
    })
    daily = fc.prepare_daily(sales)
    assert len(daily) == 2  # gaps are left for Prophet to handle


def test_train_test_split_shapes(daily):
    train, test = fc.train_test_split(daily, holdout_days=30)
    assert len(train) == 90 and len(test) == 30
    assert train["ds"].max() < test["ds"].min()


def test_train_test_split_rejects_short_series(daily):
    with pytest.raises(ValueError):
        fc.train_test_split(daily, holdout_days=200)


def test_mape():
    # (|100-90|/100 + |200-210|/200) / 2 = (10% + 5%) / 2
    assert fc.mape(np.array([100.0, 200.0]), np.array([90.0, 210.0])) == pytest.approx(7.5)
    assert fc.mape(np.array([100.0]), np.array([100.0])) == 0.0


def test_mape_excludes_zero_actuals():
    assert fc.mape(np.array([0.0, 200.0]), np.array([500.0, 180.0])) == pytest.approx(10.0)


def test_rmse():
    # errors [-1, 2] → sqrt((1 + 4) / 2)
    assert fc.rmse(np.array([2.0, 4.0]), np.array([1.0, 6.0])) == pytest.approx(np.sqrt(2.5))


def test_bias_sign():
    # predicted below actual → negative bias (under-forecasting)
    assert fc.bias(np.array([100.0]), np.array([80.0])) == pytest.approx(-20.0)


def test_metrics_nan_when_all_actuals_zero():
    assert np.isnan(fc.mape(np.array([0.0, 0.0]), np.array([1.0, 2.0])))
    assert np.isnan(fc.bias(np.array([0.0, 0.0]), np.array([1.0, 2.0])))
