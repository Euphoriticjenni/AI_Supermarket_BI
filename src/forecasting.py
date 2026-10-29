"""NexusBI - Prophet demand forecasting engine.

The module provides revenue projections, category-level forecasts, anomaly
detection, seasonal components, and a deterministic exponential-moving-
average fallback for environments where Prophet or its CmdStan backend is
not available.  The legacy helper functions at the bottom remain available
to the existing Streamlit forecasting page.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from prophet import Prophet

    _PROPHET_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - depends on the environment
    Prophet = None  # type: ignore[assignment,misc]
    _PROPHET_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CHANGEPOINT_PRIOR = 0.05
DEFAULT_INTERVAL_WIDTH = 0.90
DEFAULT_HORIZON_DAYS = 90


def _resolve_column(frame: pd.DataFrame, requested: str, aliases: tuple[str, ...]) -> Optional[str]:
    """Resolve a requested column against common legacy and new aliases."""
    if requested in frame.columns:
        return requested
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in (requested, *aliases):
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame({"ds": pd.Series(dtype="datetime64[ns]"), "y": pd.Series(dtype="float64")})


def _coerce_daily(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    fill_missing: bool,
) -> pd.DataFrame:
    """Build a clean daily ``ds``/``y`` frame from either project schema."""
    if df is None or df.empty:
        return _empty_daily()
    date_name = _resolve_column(df, date_col, ("date", "Date", "datetime", "ds"))
    value_name = _resolve_column(df, value_col, ("total_amount", "Total", "revenue", "Revenue", "y"))
    if date_name is None or value_name is None:
        logger.warning("Forecast data is missing date/value columns: date={!r}, value={!r}.", date_col, value_col)
        return _empty_daily()

    frame = pd.DataFrame({
        "ds": pd.to_datetime(df[date_name], errors="coerce").dt.normalize(),
        "y": pd.to_numeric(df[value_name], errors="coerce"),
    }).dropna(subset=["ds", "y"])
    frame["y"] = frame["y"].clip(lower=0).astype(float)
    daily = frame.groupby("ds", as_index=False)["y"].sum().sort_values("ds").reset_index(drop=True)
    if fill_missing and not daily.empty:
        full_dates = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
        daily = daily.set_index("ds").reindex(full_dates, fill_value=0.0)
        daily.index.name = "ds"
        daily = daily.reset_index()
    return daily[["ds", "y"]]


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error in percent, excluding zero actuals."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error in the source series' units."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean percentage error; negative means under-forecasting."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean((predicted[mask] - actual[mask]) / actual[mask]) * 100)


def _new_prophet(
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    seasonality_mode: str = "multiplicative",
    changepoint_prior_scale: float = DEFAULT_CHANGEPOINT_PRIOR,
) -> Any:
    """Construct Prophet and add the monthly component, with no leaked output."""
    if Prophet is None:
        raise RuntimeError(f"Prophet is unavailable: {_PROPHET_IMPORT_ERROR}")
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = Prophet(
                seasonality_mode=seasonality_mode,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                changepoint_prior_scale=changepoint_prior_scale,
                interval_width=DEFAULT_INTERVAL_WIDTH,
            )
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        return model
    except Exception as exc:
        logger.warning("Could not construct Prophet model: {}", exc)
        raise


def is_available() -> bool:
    """Return whether Prophet can construct and fit a tiny model."""
    if Prophet is None:
        return False
    try:
        model = _new_prophet(yearly_seasonality=False, weekly_seasonality=False)
        tiny = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=3), "y": [1.0, 2.0, 3.0]})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.fit(tiny)
        return True
    except Exception as exc:  # pragma: no cover - backend-specific
        logger.warning("Prophet capability probe failed; EMA fallback will be used: {}", exc)
        return False


def availability_error() -> Optional[str]:
    """Return the import/backend error when Prophet is unavailable."""
    if Prophet is None:
        return _PROPHET_IMPORT_ERROR
    if is_available():
        return None
    return "Prophet imported but its backend could not fit a model"


class DemandForecaster:
    """Fit a daily Prophet revenue model with a reliable EMA fallback."""

    def __init__(
        self,
        freq: str = "D",
        periods: int = 90,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
    ) -> None:
        self.freq = freq
        self.periods = max(int(periods), 1)
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model: Any = None
        self.history = _empty_daily()
        self._forecast = pd.DataFrame()
        self.metrics: dict[str, float] = {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
        self.training_mae = float("nan")
        self.training_rmse = float("nan")
        self.training_mape = float("nan")
        self.backend = "unfitted"

    def prepare_dataframe(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        value_col: str = "total_amount",
    ) -> pd.DataFrame:
        """Aggregate to daily dates, fill calendar gaps, and clip negatives."""
        return _coerce_daily(df, date_col, value_col, fill_missing=True)

    def _ema_forecast(self, periods: int) -> pd.DataFrame:
        """Produce a non-negative forecast when Prophet cannot be used."""
        if self.history.empty:
            return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly", "yearly"])
        y = self.history["y"].astype(float)
        span = max(2, min(30, len(y)))
        ema = float(y.ewm(span=span, adjust=False).mean().iloc[-1])
        if len(y) > 1:
            recent = y.tail(min(30, len(y)))
            slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) > 1 else 0.0
        else:
            slope = 0.0
        residual_std = float(y.diff().dropna().std()) if len(y) > 1 else 0.0
        residual_std = max(residual_std, float(y.std()) * 0.15, 1.0)
        dates = pd.date_range(self.history["ds"].max() + pd.Timedelta(days=1), periods=periods, freq=self.freq)
        step = np.arange(1, periods + 1, dtype=float)
        trend = np.maximum(0.0, ema + slope * step)
        width = residual_std * np.sqrt(step) * 1.65
        return pd.DataFrame({
            "ds": dates,
            "yhat": trend,
            "yhat_lower": np.maximum(0.0, trend - width),
            "yhat_upper": trend + width,
            "trend": trend,
            "weekly": np.zeros(periods),
            "yearly": np.zeros(periods),
        })

    @staticmethod
    def _normalise_prediction(frame: pd.DataFrame) -> pd.DataFrame:
        """Keep the public forecast schema stable across Prophet versions."""
        columns = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly", "yearly"]
        result = frame.copy()
        result["ds"] = pd.to_datetime(result["ds"], errors="coerce")
        for column in columns[1:]:
            if column not in result:
                result[column] = 0.0
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
        result["yhat"] = result["yhat"].clip(lower=0.0)
        result["yhat_lower"] = result["yhat_lower"].clip(lower=0.0)
        result["yhat_upper"] = result[["yhat_upper", "yhat", "yhat_lower"]].max(axis=1)
        return result[columns].reset_index(drop=True)

    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        value_col: str = "total_amount",
    ) -> "DemandForecaster":
        """Prepare history, fit Prophet, and calculate training metrics."""
        self.history = self.prepare_dataframe(df, date_col=date_col, value_col=value_col)
        self.model = None
        self.backend = "ema"
        if self.history.empty:
            logger.warning("DemandForecaster received no usable observations.")
            return self
        try:
            if len(self.history) < 2:
                raise ValueError("Prophet requires at least two daily observations")
            self.model = _new_prophet(
                yearly_seasonality=self.yearly_seasonality and len(self.history) >= 365,
                weekly_seasonality=self.weekly_seasonality and len(self.history) >= 14,
                daily_seasonality=self.daily_seasonality,
                seasonality_mode=self.seasonality_mode,
                changepoint_prior_scale=self.changepoint_prior_scale,
            )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.model.fit(self.history)
                fitted = self.model.predict(self.history[["ds"]])
            actual = self.history["y"].to_numpy(dtype=float)
            predicted = np.asarray(fitted["yhat"], dtype=float)
            self.metrics = {
                "mae": float(np.mean(np.abs(actual - predicted))),
                "rmse": rmse(actual, predicted),
                "mape": mape(actual, predicted),
            }
            self.training_mae = self.metrics["mae"]
            self.training_rmse = self.metrics["rmse"]
            self.training_mape = self.metrics["mape"]
            self.backend = "prophet"
            logger.info("Prophet fit complete: {} days, RMSE ${:,.2f}, MAPE {:.2f}%.", len(self.history), self.metrics["rmse"], self.metrics["mape"])
        except Exception as exc:
            self.model = None
            logger.warning("Prophet fit failed; using EMA fallback: {}", exc)
            ema = self.history["y"].ewm(span=max(2, min(30, len(self.history))), adjust=False).mean()
            actual = self.history["y"].to_numpy(dtype=float)
            predicted = ema.to_numpy(dtype=float)
            self.metrics = {"mae": float(np.mean(np.abs(actual - predicted))), "rmse": rmse(actual, predicted), "mape": mape(actual, predicted)}
            self.training_mae = self.metrics["mae"]
            self.training_rmse = self.metrics["rmse"]
            self.training_mape = self.metrics["mape"]
        return self

    def predict(self, periods: Optional[int] = None) -> pd.DataFrame:
        """Forecast future daily values, never returning negative revenue."""
        horizon = max(int(periods if periods is not None else self.periods), 1)
        if self.history.empty:
            self._forecast = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly", "yearly"])
            return self._forecast.copy()
        if self.model is None:
            self._forecast = self._ema_forecast(horizon)
            return self._forecast.copy()
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                future = self.model.make_future_dataframe(periods=horizon, freq=self.freq, include_history=False)
                forecast = self.model.predict(future)
            self._forecast = self._normalise_prediction(forecast)
            return self._forecast.copy()
        except Exception as exc:
            logger.warning("Prophet prediction failed; using EMA fallback: {}", exc)
            self._forecast = self._ema_forecast(horizon)
            return self._forecast.copy()

    def get_forecast_summary(self) -> dict[str, Any]:
        """Return horizon totals, trend direction, confidence width, and MAPE."""
        forecast = self.predict(max(self.periods, 90))
        totals = {f"next_{days}d": float(forecast.head(days)["yhat"].sum()) for days in (7, 30, 90)}
        trend_direction = "stable"
        if len(forecast) > 1:
            change = float(forecast["trend"].iloc[-1] - forecast["trend"].iloc[0])
            baseline = max(abs(float(forecast["trend"].iloc[0])), 1.0)
            trend_direction = "upward" if change / baseline > 0.01 else "downward" if change / baseline < -0.01 else "stable"
        return {
            **totals,
            "trend_direction": trend_direction,
            "confidence_width": float((forecast["yhat_upper"] - forecast["yhat_lower"]).mean()) if not forecast.empty else 0.0,
            "mape": float(self.metrics.get("mape", float("nan"))),
            "rmse": float(self.metrics.get("rmse", float("nan"))),
            "backend": self.backend,
        }

    def _history_intervals(self) -> pd.DataFrame:
        """Create actual-vs-band rows for anomaly detection."""
        if self.history.empty:
            return pd.DataFrame(columns=["ds", "actual", "yhat_lower", "yhat_upper"])
        if self.model is not None:
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    predicted = self.model.predict(self.history[["ds"]])
                return pd.DataFrame({"ds": self.history["ds"], "actual": self.history["y"], "yhat_lower": predicted["yhat_lower"].clip(lower=0), "yhat_upper": predicted["yhat_upper"]})
            except Exception as exc:
                logger.warning("Could not calculate Prophet anomaly intervals: {}", exc)
        expected = self.history["y"].ewm(span=max(2, min(30, len(self.history))), adjust=False).mean()
        spread = max(float(self.history["y"].diff().std()), float(self.history["y"].std()) * 0.15, 1.0) * 1.65
        return pd.DataFrame({"ds": self.history["ds"], "actual": self.history["y"], "yhat_lower": (expected - spread).clip(lower=0), "yhat_upper": expected + spread})

    def get_anomaly_dates(self, threshold: float = 0.05) -> pd.DataFrame:
        """Return historical dates outside the model's confidence interval."""
        intervals = self._history_intervals()
        if intervals.empty:
            return intervals.assign(anomaly_direction=pd.Series(dtype=str))
        low = intervals["yhat_lower"] * (1.0 - max(threshold, 0.0))
        high = intervals["yhat_upper"] * (1.0 + max(threshold, 0.0))
        mask = (intervals["actual"] < low) | (intervals["actual"] > high)
        result = intervals.loc[mask].copy()
        result["anomaly_direction"] = np.where(result["actual"] > result["yhat_upper"], "above_expected", "below_expected")
        return result.reset_index(drop=True)

    def get_components(self) -> dict[str, pd.DataFrame]:
        """Return trend, weekly, and yearly component frames for plotting."""
        forecast = self.predict(max(self.periods, 90))
        return {name: forecast[["ds", name]].copy() for name in ("trend", "weekly", "yearly")}


class CategoryForecaster:
    """Manage independent demand forecasters for product categories."""

    def __init__(self, categories: Optional[list[str]] = None) -> None:
        self.categories = categories
        self.forecasters: dict[str, DemandForecaster] = {}
        self.histories: dict[str, pd.DataFrame] = {}
        self._forecasts: dict[str, pd.DataFrame] = {}

    def fit_all(
        self,
        df: pd.DataFrame,
        category_col: str = "category",
        date_col: str = "date",
        value_col: str = "total_amount",
    ) -> "CategoryForecaster":
        """Fit categories with at least 60 distinct daily observations."""
        self.forecasters.clear()
        self.histories.clear()
        category_name = _resolve_column(df, category_col, ("category", "Category", "Product_Line"))
        if category_name is None:
            logger.warning("Category column {!r} not found; no category forecasts created.", category_col)
            return self
        values = self.categories or sorted(df[category_name].dropna().astype(str).unique())
        for category in values:
            subset = df.loc[df[category_name].astype(str) == str(category)]
            observed = _coerce_daily(subset, date_col, value_col, fill_missing=False)
            if len(observed) < 60:
                logger.warning("Skipping category {!r}: only {} observed days (need 60).", category, len(observed))
                continue
            prepared = _coerce_daily(subset, date_col, value_col, fill_missing=True)
            forecaster = DemandForecaster(periods=90)
            forecaster.fit(prepared, date_col="ds", value_col="y")
            self.forecasters[str(category)] = forecaster
            self.histories[str(category)] = prepared
        logger.info("Category forecasting complete: {}/{} categories fitted.", len(self.forecasters), len(values))
        return self

    def predict_all(self, periods: int = 30) -> dict[str, pd.DataFrame]:
        """Forecast every fitted category."""
        self._forecasts = {category: forecaster.predict(periods) for category, forecaster in self.forecasters.items()}
        return {category: frame.copy() for category, frame in self._forecasts.items()}

    def get_top_growth_categories(self, periods: int = 30, top_n: int = 3) -> list[dict[str, Any]]:
        """Compare historical daily averages with forecast averages."""
        forecasts = self._forecasts or self.predict_all(periods)
        results: list[dict[str, Any]] = []
        for category, forecast in forecasts.items():
            historical_avg = float(self.histories[category]["y"].tail(30).mean())
            forecast_avg = float(forecast.head(periods)["yhat"].mean()) if not forecast.empty else 0.0
            growth = (forecast_avg / historical_avg - 1.0) * 100 if historical_avg else 0.0
            results.append({"category": category, "historical_avg": historical_avg, "forecast_avg": forecast_avg, "growth_rate": growth})
        return sorted(results, key=lambda item: item["growth_rate"], reverse=True)[:max(int(top_n), 0)]


def _load_forecast_data(data_path: Optional[str | Path]) -> pd.DataFrame:
    """Load a requested path, preferring the project's processed artefacts."""
    if data_path:
        path = Path(data_path)
    else:
        candidates = [config.PROCESSED_DATA_PATH / "sales_features.parquet", config.PROCESSED_DATA_PATH / "sales_processed.parquet", config.SALES_DATA_FILE]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    if not path.exists():
        logger.warning("Forecast data path does not exist: {}", path)
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not load forecast data from {}: {}", path, exc)
        return pd.DataFrame()


def run_forecast_pipeline(data_path: Optional[str | Path] = None, periods: int = 90) -> dict[str, Any]:
    """Load data, fit total/category forecasts, and return all forecast outputs."""
    empty_result: dict[str, Any] = {"total_forecast": pd.DataFrame(), "category_forecasts": {}, "summary": {}, "anomalies": pd.DataFrame(), "components": {}}
    try:
        data = _load_forecast_data(data_path)
        if data.empty:
            logger.warning("Forecast pipeline has no input data; returning empty result.")
            return empty_result
        total = DemandForecaster(periods=periods).fit(data)
        total_forecast = total.predict(periods)
        categories = CategoryForecaster().fit_all(data)
        category_forecasts = categories.predict_all(periods)
        summary = total.get_forecast_summary()
        summary["top_growth_categories"] = categories.get_top_growth_categories(periods=periods)
        result = {
            "total_forecast": total_forecast,
            "category_forecasts": category_forecasts,
            "summary": summary,
            "anomalies": total.get_anomaly_dates(),
            "components": total.get_components(),
        }
        logger.info("Forecast pipeline complete: {} forecast days, {} category forecasts, {} anomalies.", len(total_forecast), len(category_forecasts), len(result["anomalies"]))
        return result
    except Exception as exc:
        logger.exception("Forecast pipeline failed; returning a usable empty result: {}", exc)
        return empty_result


def generate_forecast_insights(forecast: dict[str, Any]) -> list[str]:
    """Convert forecast outputs into five to seven plain-English insights."""
    summary = forecast.get("summary", {}) or {}
    insights: list[str] = []
    if summary.get("next_30d") is not None:
        insights.append(f"Expected revenue for the next 30 days is ${float(summary['next_30d']):,.0f}.")
    if summary.get("trend_direction"):
        insights.append(f"The underlying revenue trend is {summary['trend_direction']}.")
    top_growth = summary.get("top_growth_categories", []) or []
    for item in top_growth[:3]:
        insights.append(f"{item['category']} is the fastest-growing category, at {item['growth_rate']:+.1f}% versus its recent daily average.")
    anomalies = forecast.get("anomalies")
    if isinstance(anomalies, pd.DataFrame):
        insights.append(f"The historical model flags {len(anomalies):,} unusual revenue dates for review.")
    if summary.get("confidence_width") is not None:
        insights.append(f"The average daily forecast confidence width is ${float(summary['confidence_width']):,.0f}; treat the estimate as a planning range.")
    if summary.get("mape") is not None and not pd.isna(summary.get("mape")):
        insights.append(f"Training MAPE is {float(summary['mape']):.1f}%; validate the forecast against current promotions and supply constraints.")
    defaults = [
        "Use the forecast as a planning baseline and refresh it after major promotions or assortment changes.",
        "Review category forecasts alongside inventory lead times before placing replenishment orders.",
        "Investigate flagged dates against returns, voids, promotions, and unusual bulk orders.",
        "Compare forecast confidence ranges with available cash and storage capacity when budgeting purchases.",
        "Track forecast error weekly so the planning team can adjust assumptions before the next ordering cycle.",
    ]
    for default in defaults:
        if len(insights) >= 5:
            break
        insights.append(default)
    return insights[:7]


# ---------------------------------------------------------------------------
# Backward-compatible helpers used by the existing dashboard and tests.
# ---------------------------------------------------------------------------
def prepare_daily(sales_df: pd.DataFrame, date_col: str = "Date", value_col: str = "Total") -> pd.DataFrame:
    """Aggregate transactions to daily ``ds``/``y`` without filling gaps."""
    return _coerce_daily(sales_df, date_col, value_col, fill_missing=False)


def train_test_split(daily: pd.DataFrame, holdout_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the final ``holdout_days`` rows from a daily series."""
    if holdout_days <= 0 or len(daily) <= holdout_days:
        raise ValueError(f"need more than {holdout_days} daily observations for a holdout split, got {len(daily)}")
    split = len(daily) - holdout_days
    return daily.iloc[:split].copy(), daily.iloc[split:].copy()


def _build_model(daily: pd.DataFrame, changepoint_prior_scale: float = DEFAULT_CHANGEPOINT_PRIOR, interval_width: float = DEFAULT_INTERVAL_WIDTH) -> Any:
    """Legacy Prophet model builder retained for callers outside this module."""
    model = _new_prophet(yearly_seasonality=len(daily) >= 540, weekly_seasonality=len(daily) >= 21, changepoint_prior_scale=changepoint_prior_scale)
    model.interval_width = interval_width
    return model


def evaluate_backtest(daily: pd.DataFrame, holdout_days: int = 30, changepoint_prior_scale: float = DEFAULT_CHANGEPOINT_PRIOR, interval_width: float = DEFAULT_INTERVAL_WIDTH) -> dict[str, Any]:
    """Evaluate Prophet or an EMA fallback on the final holdout window."""
    try:
        train, test = train_test_split(daily, holdout_days)
    except Exception as exc:
        logger.warning("Backtest split failed: {}", exc)
        return {"mape": float("nan"), "rmse": float("nan"), "bias": float("nan"), "coverage": 0.0, "holdout_days": holdout_days, "eval_df": pd.DataFrame()}
    forecaster = DemandForecaster(periods=holdout_days, changepoint_prior_scale=changepoint_prior_scale).fit(train, date_col="ds", value_col="y")
    predicted = forecaster.predict(holdout_days)
    actual = test["y"].to_numpy(dtype=float)
    prediction = predicted["yhat"].to_numpy(dtype=float)[:len(actual)]
    lower = predicted["yhat_lower"].to_numpy(dtype=float)[:len(actual)]
    upper = predicted["yhat_upper"].to_numpy(dtype=float)[:len(actual)]
    eval_df = pd.DataFrame({"ds": test["ds"].to_numpy(), "actual": actual, "predicted": prediction, "yhat_lower": lower, "yhat_upper": upper})
    return {"mape": mape(actual, prediction), "rmse": rmse(actual, prediction), "bias": bias(actual, prediction), "coverage": float(np.mean((actual >= lower) & (actual <= upper))), "holdout_days": int(holdout_days), "eval_df": eval_df}


def fit_forecast(daily: pd.DataFrame, periods: int = 30, changepoint_prior_scale: float = DEFAULT_CHANGEPOINT_PRIOR, interval_width: float = DEFAULT_INTERVAL_WIDTH) -> dict[str, Any]:
    """Legacy wrapper returning the dashboard's history/forecast dictionary."""
    forecaster = DemandForecaster(periods=periods, changepoint_prior_scale=changepoint_prior_scale).fit(daily, date_col="ds", value_col="y")
    return {"history": forecaster.history.copy(), "forecast": forecaster.predict(periods), "params": {"periods": int(periods), "changepoint_prior_scale": float(changepoint_prior_scale), "interval_width": float(interval_width)}, "last_observed": forecaster.history["ds"].max() if not forecaster.history.empty else pd.NaT}


if __name__ == "__main__":
    result = run_forecast_pipeline()
    print(generate_forecast_insights(result))
