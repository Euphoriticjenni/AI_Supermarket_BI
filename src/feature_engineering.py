"""Feature engineering for the AI_Supermarket_BI project.

Enriches the processed sales data with model-ready features:

* **Temporal** — purchase hour, day-of-week/month/quarter/year names,
  weekend flag and a month-day holiday flag (derived from the calendar in
  ``config.HOLIDAYS``: New Year, Valentine's, Easter, Labour Day,
  Thingyan, Halloween, Christmas, New Year's Eve ...).
* **Financial** — profit, profit margin and effective tax rate.
* **Rolling** — 7-day and 30-day time-based rolling means of ``Total``
  computed per product line, forward-filled then zero-filled.
* **Lag** — daily total sales per product line lagged by 1, 7 and 30 days
  on a complete daily grid (gaps filled as missing days).
* **Basket** — items per transaction and average basket value per invoice.

Public entry point: :func:`run_feature_engineering`.
"""
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

#: Major holidays as (month, day) pairs — year-independent, derived from the
#: per-year calendar in config.HOLIDAYS (Easter dates vary, so the config
#: list carries one entry per year and is collapsed into month-day pairs).
_HOLIDAY_MONTH_DAY = {
    (int(iso[5:7]), int(iso[8:10])) for iso in config.HOLIDAYS
}


# ---------------------------------------------------------------------------
# Feature builders (each takes and returns a DataFrame)
# ---------------------------------------------------------------------------
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from ``Date`` and ``Time``."""
    try:
        dates = df["Date"]
        if "Time" in df.columns:
            df["Purchase_Hour"] = pd.to_datetime(df["Time"], format="%H:%M").dt.hour.astype("int8")
        df["Day_of_Week"] = dates.dt.dayofweek.astype("int8")
        df["Day_Name"] = dates.dt.day_name()
        df["Month"] = dates.dt.month.astype("int8")
        df["Month_Name"] = dates.dt.month_name()
        df["Quarter"] = dates.dt.quarter.astype("int8")
        df["Year"] = dates.dt.year.astype("int16")
        df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(bool)
        df["Is_Holiday"] = [
            (month, day) in _HOLIDAY_MONTH_DAY
            for month, day in zip(dates.dt.month, dates.dt.day)
        ]
        logger.info(
            "Temporal features added ({:,} weekend rows, {:,} holiday rows).",
            int(df["Is_Weekend"].sum()), int(df["Is_Holiday"].sum()),
        )
        return df
    except Exception:
        logger.exception("add_temporal_features failed")
        raise


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Profit, Profit_Margin and the effective Tax_Rate (percent)."""
    try:
        df["Profit"] = (df["Total"] - df["COGS"]).round(2)
        df["Profit_Margin"] = np.where(
            df["Total"] > 0, df["Profit"] / df["Total"] * 100, 0.0
        ).round(4)
        df["Tax_Rate"] = np.where(
            df["COGS"] > 0, df["Tax_5pct"] / df["COGS"] * 100, 0.0
        ).round(4)
        logger.info(
            "Financial features added (profit ${:,.2f}, mean margin {:.2f}%, mean tax {:.2f}%).",
            df["Profit"].sum(), df["Profit_Margin"].mean(), df["Tax_Rate"].mean(),
        )
        return df
    except Exception:
        logger.exception("add_financial_features failed")
        raise


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-product-line 7-day and 30-day rolling means of daily sales.

    Mirrors :func:`add_lag_features`: daily ``Total`` revenue is aggregated
    per product line on a complete daily grid, a time-based rolling mean is
    computed, and the result is mapped back onto every transaction of that
    date and line (so a row's value = mean daily revenue for its line over
    the trailing window). Residual NaNs are forward-filled then zero-filled.
    """
    try:
        df = df.sort_values("Date").reset_index(drop=True)
        pivot = _daily_per_product_line(df)

        stacked = [pivot.rolling(window=w, min_periods=1).mean().stack(dropna=False)
                   .rename(c) for c, w in
                   (("Rolling_7d_Sales", "7D"), ("Rolling_30d_Sales", "30D"))]
        rolled = pd.concat(stacked, axis=1).reset_index()
        rolled = rolled.rename(columns={rolled.columns[0]: "_r_date",
                                        rolled.columns[1]: "_r_line"})
        rolled["_r_line"] = rolled["_r_line"].astype(str)

        df = df.assign(_date=df["Date"].dt.normalize(), _line=df["Product_Line"].astype(str))
        df = df.merge(rolled, how="left", left_on=["_date", "_line"],
                      right_on=["_r_date", "_r_line"])
        df = df.drop(columns=["_date", "_line", "_r_date", "_r_line"], errors="ignore")

        for column in ("Rolling_7d_Sales", "Rolling_30d_Sales"):
            df[column] = df[column].round(2).ffill().fillna(0.0)
        logger.info(
            "Rolling features added (7d mean ${:,.2f}, 30d mean ${:,.2f} per line-day).",
            df["Rolling_7d_Sales"].mean(), df["Rolling_30d_Sales"].mean(),
        )
        return df
    except Exception:
        logger.exception("add_rolling_features failed")
        raise


def _daily_per_product_line(df: pd.DataFrame) -> pd.DataFrame:
    """Return a complete date x product-line grid of daily Total sales."""
    daily = (
        df.groupby([df["Date"].dt.normalize().rename("Date"), "Product_Line"], observed=True)["Total"]
        .sum()
        .reset_index()
    )
    full_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    pivot = daily.pivot(index="Date", columns="Product_Line", values="Total").reindex(full_dates)
    return pivot


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-product-line daily sales lagged by 1, 7 and 30 days.

    The daily totals are computed on a complete daily grid per product line
    (calendar gaps become NaN days) so ``shift(k)`` always means "k
    calendar days ago".
    """
    try:
        pivot = _daily_per_product_line(df)
        lags = pd.concat(
            [pivot.shift(k).stack(dropna=False).rename(f"Sales_Lag_{k}") for k in (1, 7, 30)],
            axis=1,
        ).reset_index()
        # The grid reindex leaves the date level unnamed; rename by position
        # so the merge keys are always well-defined.
        lags = lags.rename(columns={lags.columns[0]: "_lag_date"})
        lags = lags.rename(columns={lags.columns[1]: "_lag_line"})
        lags["_lag_line"] = lags["_lag_line"].astype(str)

        df = df.assign(_date=df["Date"].dt.normalize(), _line=df["Product_Line"].astype(str))
        df = df.merge(
            lags,
            how="left",
            left_on=["_date", "_line"],
            right_on=["_lag_date", "_lag_line"],
        )
        df = df.drop(columns=["_date", "_line", "_lag_date", "_lag_line"], errors="ignore")

        n_missing = int(df["Sales_Lag_1"].isna().sum())
        logger.info(
            "Lag features added ({} rows lack Sales_Lag_1 — first day per line).", n_missing,
        )
        return df
    except Exception:
        logger.exception("add_lag_features failed")
        raise


def add_basket_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basket features.

    * ``Items_Per_Transaction`` — the line quantity (the sales data is
      invoice-level, one product line per invoice).
    * ``Avg_Basket_Value`` — mean ``Total`` per ``Invoice_ID``. With unique
      invoice IDs this equals the row's own ``Total``; the mapping is kept
      explicit so multi-line invoice data would work unchanged.
    """
    try:
        df["Items_Per_Transaction"] = df["Quantity"].astype("int64")
        invoice_means = df.groupby("Invoice_ID", observed=True)["Total"].mean()
        df["Avg_Basket_Value"] = df["Invoice_ID"].map(invoice_means).round(2)
        logger.info(
            "Basket features added (mean basket value ${:,.2f} across {:,} invoices).",
            invoice_means.mean(), len(invoice_means),
        )
        return df
    except Exception:
        logger.exception("add_basket_features failed")
        raise


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_feature_engineering(df: Optional[pd.DataFrame] = None, save: bool = True) -> pd.DataFrame:
    """Run the full feature-engineering chain.

    Parameters
    ----------
    df:
        A cleaned sales DataFrame. When omitted, the processed sales file
        from :mod:`src.preprocessing` is loaded.
    save:
        When True (default) the enriched frame is persisted as
        ``sales_features.parquet`` (CSV fallback) for the dashboard.

    Returns
    -------
    pandas.DataFrame
        The enriched DataFrame.
    """
    logger.info("=== Feature engineering started ===")
    try:
        if df is None:
            from src.preprocessing import load_processed_or_csv
            df = load_processed_or_csv()
        base_columns = set(df.columns)

        df = add_temporal_features(df)
        df = add_financial_features(df)
        df = add_basket_features(df)
        df = add_rolling_features(df)
        df = add_lag_features(df)

        added = [c for c in df.columns if c not in base_columns]
        if save:
            from src.preprocessing import save_processed_data
            save_processed_data(df, "sales_features")
        logger.info(
            "=== Feature engineering complete: {} features added -> {} total columns ===",
            len(added), df.shape[1],
        )
        logger.info("New features: {}", ", ".join(added))
        return df
    except Exception:
        logger.exception("Feature engineering pipeline failed")
        raise


if __name__ == "__main__":
    run_feature_engineering()
