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
from __future__ import annotations

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
# Column resolution (supports both legacy and new schemas)
# ---------------------------------------------------------------------------
def _resolve_column(frame: pd.DataFrame, requested: str, aliases: tuple[str, ...]) -> Optional[str]:
    """Resolve a requested column against common legacy and new aliases."""
    if requested in frame.columns:
        return requested
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in (requested, *aliases):
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical internal names to actual DataFrame columns."""
    return {
        "date": _resolve_column(df, "Date", ("date", "datetime", "ds")),
        "time": _resolve_column(df, "Time", ("time",)),
        "total": _resolve_column(df, "Total", ("total_amount", "Total", "revenue", "Revenue", "y")),
        "cogs": _resolve_column(df, "COGS", ("unit_cost", "COGS", "cost")),
        "tax": _resolve_column(df, "Tax_5pct", ("tax", "Tax_5pct")),
        "invoice_id": _resolve_column(df, "Invoice_ID", ("transaction_id", "Invoice_ID", "invoice_id")),
        "product_line": _resolve_column(df, "Product_Line", ("category", "Product_Line", "product_line")),
        "quantity": _resolve_column(df, "Quantity", ("quantity", "Quantity")),
        "unit_price": _resolve_column(df, "Unit_Price", ("unit_price", "Unit_Price", "Selling_Price")),
        "unit_cost": _resolve_column(df, "Cost", ("unit_cost", "Cost", "unit_cost")),
        "gross_income": _resolve_column(df, "Gross_Income", ("profit", "Gross_Income", "profit")),
        "gross_margin_pct": _resolve_column(df, "Gross_Margin_Pct", ("margin", "Gross_Margin_Pct")),
    }


# ---------------------------------------------------------------------------
# Feature builders (each takes and returns a DataFrame)
# ---------------------------------------------------------------------------
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from ``Date`` and ``Time``."""
    try:
        cols = _resolve_columns(df)
        date_col = cols["date"]
        time_col = cols["time"]
        if date_col is None:
            logger.warning("No date column found; skipping temporal features.")
            return df

        dates = pd.to_datetime(df[date_col], errors="coerce")
        if time_col and time_col in df.columns:
            # Support both "HH:MM" and "HH:MM:SS" time strings
            parsed_time = pd.to_datetime(df[time_col].astype(str), errors="coerce")
            df["Purchase_Hour"] = parsed_time.dt.hour.fillna(0).astype("int8")
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
        cols = _resolve_columns(df)
        total_col = cols["total"]
        cogs_col = cols["cogs"]
        tax_col = cols["tax"]
        gross_income_col = cols["gross_income"]
        gross_margin_col = cols["gross_margin_pct"]

        # Profit: prefer existing profit/gross_income, else compute from Total - COGS
        if gross_income_col and gross_income_col in df.columns:
            df["Profit"] = pd.to_numeric(df[gross_income_col], errors="coerce").round(2)
        elif total_col and cogs_col and total_col in df.columns and cogs_col in df.columns:
            df["Profit"] = (pd.to_numeric(df[total_col], errors="coerce") - pd.to_numeric(df[cogs_col], errors="coerce")).round(2)
        else:
            df["Profit"] = 0.0

        # Profit_Margin
        if gross_margin_col and gross_margin_col in df.columns:
            df["Profit_Margin"] = pd.to_numeric(df[gross_margin_col], errors="coerce").round(4)
        elif total_col and total_col in df.columns:
            total_vals = pd.to_numeric(df[total_col], errors="coerce")
            df["Profit_Margin"] = np.where(
                total_vals > 0, df["Profit"] / total_vals * 100, 0.0
            ).round(4)
        else:
            df["Profit_Margin"] = 0.0

        # Tax_Rate
        if tax_col and tax_col in df.columns and cogs_col and cogs_col in df.columns:
            cogs_vals = pd.to_numeric(df[cogs_col], errors="coerce")
            tax_vals = pd.to_numeric(df[tax_col], errors="coerce")
            df["Tax_Rate"] = np.where(
                cogs_vals > 0, tax_vals / cogs_vals * 100, 0.0
            ).round(4)
        else:
            df["Tax_Rate"] = 0.0

        logger.info(
            "Financial features added (profit ${:,.2f}, mean margin {:.2f}%, mean tax {:.2f}%).",
            df["Profit"].sum(), df["Profit_Margin"].mean(), df["Tax_Rate"].mean(),
        )
        return df
    except Exception:
        logger.exception("add_financial_features failed")
        raise


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-product-line 7-day and 30-day rolling means of daily sales."""
    try:
        cols = _resolve_columns(df)
        date_col = cols["date"]
        total_col = cols["total"]
        product_line_col = cols["product_line"]

        if not all([date_col, total_col, product_line_col]):
            logger.warning("Missing columns for rolling features; skipping.")
            return df

        df = df.sort_values(date_col).reset_index(drop=True)
        pivot = _daily_per_product_line(df, date_col, total_col, product_line_col)

        stacked = [
            pivot.rolling(window=w, min_periods=1).mean().stack(dropna=False).rename(c)
            for c, w in (("Rolling_7d_Sales", "7D"), ("Rolling_30d_Sales", "30D"))
        ]
        rolled = pd.concat(stacked, axis=1).reset_index()
        rolled = rolled.rename(columns={rolled.columns[0]: "_r_date", rolled.columns[1]: "_r_line"})
        rolled["_r_line"] = rolled["_r_line"].astype(str)

        df = df.assign(_date=pd.to_datetime(df[date_col]).dt.normalize(), _line=df[product_line_col].astype(str))
        df = df.merge(rolled, how="left", left_on=["_date", "_line"], right_on=["_r_date", "_r_line"])
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


def _daily_per_product_line(
    df: pd.DataFrame, date_col: str, total_col: str, product_line_col: str
) -> pd.DataFrame:
    """Return a complete date x product-line grid of daily Total sales."""
    daily = (
        df.groupby(
            [pd.to_datetime(df[date_col]).dt.normalize().rename("Date"), product_line_col],
            observed=True,
        )[total_col]
        .sum()
        .reset_index()
    )
    full_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    pivot = daily.pivot(index="Date", columns=product_line_col, values=total_col).reindex(full_dates)
    return pivot


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-product-line daily sales lagged by 1, 7 and 30 days."""
    try:
        cols = _resolve_columns(df)
        date_col = cols["date"]
        total_col = cols["total"]
        product_line_col = cols["product_line"]

        if not all([date_col, total_col, product_line_col]):
            logger.warning("Missing columns for lag features; skipping.")
            return df

        pivot = _daily_per_product_line(df, date_col, total_col, product_line_col)
        lags = pd.concat(
            [pivot.shift(k).stack(dropna=False).rename(f"Sales_Lag_{k}") for k in (1, 7, 30)],
            axis=1,
        ).reset_index()
        lags = lags.rename(columns={lags.columns[0]: "_lag_date", lags.columns[1]: "_lag_line"})
        lags["_lag_line"] = lags["_lag_line"].astype(str)

        df = df.assign(_date=pd.to_datetime(df[date_col]).dt.normalize(), _line=df[product_line_col].astype(str))
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
        cols = _resolve_columns(df)
        quantity_col = cols["quantity"]
        total_col = cols["total"]
        invoice_id_col = cols["invoice_id"]

        if quantity_col and quantity_col in df.columns:
            df["Items_Per_Transaction"] = pd.to_numeric(df[quantity_col], errors="coerce").astype("int64")
        else:
            df["Items_Per_Transaction"] = 1

        if invoice_id_col and total_col and invoice_id_col in df.columns and total_col in df.columns:
            invoice_means = df.groupby(invoice_id_col, observed=True)[total_col].mean()
            df["Avg_Basket_Value"] = df[invoice_id_col].map(invoice_means).round(2)
            logger.info(
                "Basket features added (mean basket value ${:,.2f} across {:,} invoices).",
                invoice_means.mean(), len(invoice_means),
            )
        else:
            df["Avg_Basket_Value"] = 0.0
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