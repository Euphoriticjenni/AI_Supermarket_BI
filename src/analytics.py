"""Lightweight analytics for the AI_Supermarket_BI project.

Pure pandas + numpy + plotly — no ML libraries anywhere. Provides:

* :func:`rfm_analysis` — RFM customer scoring with a deterministic customer
  proxy, rule-based segments and a simple CLV model.
* :func:`product_analysis` — best sellers, slow movers, profitability,
  seasonal demand, ABC classification + charts.
* :func:`trend_analysis` — daily/weekly/monthly trends, YoY, moving
  averages, growth rates and a seasonal-adjusted 30-day projection.
* :func:`run_full_analytics` — chains everything and persists a summary.

All plotly figures are returned as objects only (rendered by Streamlit);
static PNG export is deliberately NOT done here — kaleido is broken on
this host, and the PDF report uses matplotlib instead.
"""
from __future__ import annotations

import zlib
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

#: Shared palette (Set2) — same family as src/visualization.py.
PALETTE = px.colors.qualitative.Set2
PALETTE_HEX = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]

_TEMPLATE = dict(template=config.PLOTLY_TEMPLATE, height=config.CHART_HEIGHT)


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
        "branch": _resolve_column(df, "Branch", ("branch", "Branch")),
        "city": _resolve_column(df, "City", ("city", "City")),
        "customer_type": _resolve_column(df, "Customer_Type", ("membership_tier", "Customer_Type", "customer_type")),
        "gender": _resolve_column(df, "Gender", ("gender", "Gender")),
        "payment": _resolve_column(df, "Payment", ("payment_method", "Payment", "payment_method")),
        "rating": _resolve_column(df, "Rating", ("rating", "Rating")),
    }


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------
def _score_1_to_5(series: pd.Series, invert: bool = False) -> pd.Series:
    """Quantile-score a series into 1..5 (5 = "best").

    Uses ``pd.qcut`` first; when binning fails (heavy duplicates /
    constant series) falls back to ``rank(pct=True)`` mapped manually to
    five bins, exactly as required by the pipeline spec.

    Parameters
    ----------
    series:
        The values to score.
    invert:
        When True the scale is flipped so that SMALL values get the high
        scores (used for Recency, where fewer days since last purchase is
        better).
    """
    try:
        scores = pd.qcut(series, q=5, labels=False, duplicates="raise") + 1
    except ValueError:
        pct = series.rank(pct=True, method="first")
        scores = np.ceil(pct * 5).clip(1, 5)
    scores = scores.astype(int)
    if invert:
        scores = 6 - scores
    return scores


def _qcut_labels(series: pd.Series, labels: list) -> pd.Series:
    """Bin a series into len(labels) tiers (Low..High), duplicate-safe."""
    try:
        return pd.qcut(series, q=len(labels), labels=labels, duplicates="raise")
    except ValueError:
        pct = series.rank(pct=True, method="first")
        edges = np.linspace(0, 1, len(labels) + 1)
        codes = np.clip(np.digitize(pct, edges[1:-1]), 0, len(labels) - 1)
        return pd.Series(np.array(labels)[codes], index=series.index)


# ---------------------------------------------------------------------------
# RFM analysis
# ---------------------------------------------------------------------------
def rfm_analysis(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM scores, rule-based segments and CLV per proxy customer.

    The sales table has no customer key, so a deterministic customer proxy
    is derived from the invoice ID::

        Customer_ID = "CUST-" + f"{zlib.crc32(Invoice_ID) % 10000:05d}"

    (``zlib.crc32`` is used instead of the built-in ``hash()`` because
    Python salts string hashes per process — crc32 keeps the mapping
    stable across processes, so dashboard reloads reproduce the same
    customers.)

    Recency is scored inverted (5 = most recent purchase) so that every
    score reads "higher is better", matching the segment rules below.

    Returns
    -------
    pandas.DataFrame
        Columns: Customer_ID, Recency, Frequency, Monetary, R_Score,
        F_Score, M_Score, RFM_Segment, CLV, CLV_Tier.
    """
    logger.info("RFM analysis started for {:,} sales rows ...", len(sales_df))
    cols = _resolve_columns(sales_df)
    df = sales_df.copy()

    date_col = cols["date"]
    invoice_id_col = cols["invoice_id"]
    total_col = cols["total"]

    if not date_col or not invoice_id_col or not total_col:
        logger.warning("Missing required columns for RFM; returning empty result.")
        return pd.DataFrame(columns=[
            "Customer_ID", "Recency", "Frequency", "Monetary",
            "R_Score", "F_Score", "M_Score", "RFM_Segment", "CLV", "CLV_Tier",
        ])

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Deterministic customer proxy from the invoice ID.
    df["Customer_ID"] = "CUST-" + df[invoice_id_col].map(
        lambda inv: f"{zlib.crc32(str(inv).encode('utf-8')) % 10000:05d}"
    )

    snapshot = df[date_col].max()
    grouped = df.groupby("Customer_ID", observed=True).agg(
        Last_Purchase=(date_col, "max"),
        Frequency=(invoice_id_col, "nunique"),
        Monetary=(total_col, "sum"),
    ).reset_index()
    grouped["Recency"] = (snapshot - grouped["Last_Purchase"]).dt.days
    grouped = grouped[["Customer_ID", "Recency", "Frequency", "Monetary"]]

    # Scores 1-5: R inverted (recent = 5), F and M direct.
    grouped["R_Score"] = _score_1_to_5(grouped["Recency"], invert=True)
    grouped["F_Score"] = _score_1_to_5(grouped["Frequency"])
    grouped["M_Score"] = _score_1_to_5(grouped["Monetary"])

    # Rule-based segments (pure conditionals, evaluated top-down).
    r, f, m = grouped["R_Score"], grouped["F_Score"], grouped["M_Score"]
    grouped["RFM_Segment"] = np.select(
        [
            (r >= 4) & (f >= 4) & (m >= 4),
            (r >= 3) & (f >= 3) & (m >= 3),
            (r >= 4) & (f <= 2),
            (r <= 2) & (f >= 3) & (m >= 3),
            (r <= 2) & (f <= 2) & (m <= 2),
        ],
        [
            "Champions",
            "Loyal",
            "New Customers",
            "At Risk",
            "Lost",
        ],
        default="Potential Loyalists",
    )

    # Simple CLV: average order value x frequency x annualised recency factor.
    frequency_safe = grouped["Frequency"].clip(lower=1)
    recency_factor = (365.0 / grouped["Recency"].clip(lower=1)).clip(lower=1)
    grouped["CLV"] = (
        (grouped["Monetary"] / frequency_safe) * frequency_safe * recency_factor
    ).round(2)
    grouped["CLV_Tier"] = _qcut_labels(grouped["CLV"], ["Low", "Medium", "High"])

    result = grouped[[
        "Customer_ID", "Recency", "Frequency", "Monetary",
        "R_Score", "F_Score", "M_Score", "RFM_Segment", "CLV", "CLV_Tier",
    ]]
    logger.info(
        "RFM complete: {:,} proxy customers; segments: {}",
        len(result), result["RFM_Segment"].value_counts().to_dict(),
    )
    return result


# ---------------------------------------------------------------------------
# Product analysis
# ---------------------------------------------------------------------------
def product_analysis(sales_df: pd.DataFrame) -> Dict:
    """Product-line level analysis: sellers, profitability, seasonality, ABC.

    Returns
    -------
    dict
        ``best_sellers``, ``slow_movers``, ``profitability``, ``seasonal``
        (Month x Product_Line pivot), ``abc`` (with cumulative % and class)
        and ``charts`` (dict of plotly figures).
    """
    logger.info("Product analysis started ...")
    cols = _resolve_columns(sales_df)
    df = sales_df.copy()

    date_col = cols["date"]
    total_col = cols["total"]
    invoice_id_col = cols["invoice_id"]
    product_line_col = cols["product_line"]
    gross_income_col = cols["gross_income"]

    if not all([date_col, total_col, invoice_id_col, product_line_col]):
        logger.warning("Missing required columns for product analysis; returning empty result.")
        empty_df = pd.DataFrame()
        return {
            "best_sellers": empty_df,
            "slow_movers": empty_df,
            "profitability": empty_df,
            "seasonal": empty_df,
            "abc": empty_df,
            "charts": {},
        }

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Month_Num"] = df[date_col].dt.month

    line_stats = df.groupby(product_line_col, observed=True).agg(
        Revenue=(total_col, "sum"),
        Orders=(invoice_id_col, "nunique"),
    ).reset_index()
    # Normalize output column to "Product_Line" for downstream consumers
    line_stats = line_stats.rename(columns={product_line_col: "Product_Line"})
    line_stats["Avg_Order"] = (line_stats["Revenue"] / line_stats["Orders"]).round(2)

    best_sellers = line_stats.sort_values("Revenue", ascending=False).head(10).reset_index(drop=True)
    slow_movers = line_stats.sort_values("Revenue").head(5).reset_index(drop=True)

    # Profitability: use gross_income if available, else estimate from total * margin
    if gross_income_col and gross_income_col in df.columns:
        profit_col = gross_income_col
    else:
        # Estimate profit as a fraction of total (using Gross_Margin_Pct default ~4.76%)
        margin_col = cols["gross_margin_pct"]
        if margin_col and margin_col in df.columns:
            df["_profit_est"] = pd.to_numeric(df[total_col], errors="coerce") * pd.to_numeric(df[margin_col], errors="coerce") / 100
            profit_col = "_profit_est"
        else:
            df["_profit_est"] = pd.to_numeric(df[total_col], errors="coerce") * 0.0476
            profit_col = "_profit_est"

    profitability = (
        df.groupby(product_line_col, observed=True)
        .agg(Revenue=(total_col, "sum"), Profit=(profit_col, "sum"))
        .reset_index()
    )
    profitability = profitability.rename(columns={product_line_col: "Product_Line"})
    profitability["Margin_Pct"] = (profitability["Profit"] / profitability["Revenue"] * 100).round(2)
    profitability = profitability.sort_values("Revenue", ascending=False).reset_index(drop=True)

    seasonal = df.pivot_table(
        index="Month_Num", columns=product_line_col, values=total_col,
        aggfunc="sum", observed=True,
    ).fillna(0.0)
    seasonal.index.name = "Month"

    # ABC classification on cumulative revenue share.
    abc = profitability[["Product_Line", "Revenue"]].copy()
    abc["Revenue_Pct"] = (abc["Revenue"] / abc["Revenue"].sum() * 100).round(2)
    abc["Cumulative_Pct"] = abc["Revenue_Pct"].cumsum().round(2)
    abc["ABC_Class"] = np.where(
        abc["Cumulative_Pct"] <= 80.0, "A",
        np.where(abc["Cumulative_Pct"] <= 95.0, "B", "C"),
    )

    # ---- charts (plotly objects only — never exported here) -----------
    charts: Dict = {}

    best_sellers_chart = px.bar(
        best_sellers.sort_values("Revenue"),
        x="Revenue", y="Product_Line", orientation="h",
        color="Product_Line", color_discrete_sequence=PALETTE,
        title="Best-Selling Product Lines (by revenue)",
        labels={"Revenue": "Revenue ($)", "Product_Line": "Product Line"},
        hover_data=["Orders", "Avg_Order"],
    )
    best_sellers_chart.update_layout(**_TEMPLATE, showlegend=False)
    charts["best_sellers_chart"] = best_sellers_chart

    profit_long = profitability.melt(
        id_vars="Product_Line", value_vars=["Revenue", "Profit"],
        var_name="Measure", value_name="Amount",
    )
    profitability_chart = px.bar(
        profit_long, x="Product_Line", y="Amount", color="Measure",
        barmode="group", color_discrete_sequence=PALETTE[:2],
        title="Revenue vs Profit by Product Line",
        labels={"Amount": "Amount ($)", "Product_Line": "Product Line"},
    )
    profitability_chart.update_layout(**_TEMPLATE, xaxis_tickangle=-20)
    charts["profitability_chart"] = profitability_chart

    abc_chart = make_subplots(specs=[[{"secondary_y": True}]])
    abc_chart.add_trace(
        go.Bar(
            x=abc["Product_Line"], y=abc["Revenue"], name="Revenue",
            marker_color=PALETTE_HEX[0],
        ),
        secondary_y=False,
    )
    abc_chart.add_trace(
        go.Scatter(
            x=abc["Product_Line"], y=abc["Cumulative_Pct"], name="Cumulative %",
            mode="lines+markers", line=dict(color=PALETTE_HEX[1], dash="dash"),
        ),
        secondary_y=True,
    )
    abc_chart.update_layout(
        title="ABC Classification — Revenue with Cumulative Share",
        template=config.PLOTLY_TEMPLATE, height=config.CHART_HEIGHT,
    )
    abc_chart.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    abc_chart.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
    abc_chart.update_xaxes(tickangle=-20)
    charts["abc_chart"] = abc_chart

    logger.info(
        "Product analysis complete: {} lines, top = {} (${:.0f}), class A = {} lines",
        len(profitability), best_sellers.iloc[0]["Product_Line"],
        best_sellers.iloc[0]["Revenue"], int((abc["ABC_Class"] == "A").sum()),
    )
    return {
        "best_sellers": best_sellers,
        "slow_movers": slow_movers,
        "profitability": profitability,
        "seasonal": seasonal,
        "abc": abc,
        "charts": charts,
    }


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------
def trend_analysis(sales_df: pd.DataFrame) -> Dict:
    """Trend, seasonality and projection analysis.

    Returns
    -------
    dict
        ``daily_trend``, ``weekly_trend``, ``monthly_trend``,
        ``yoy_comparison`` (pivot Month x Year), ``moving_averages``,
        ``growth_rates`` (MoM / YoY), ``projection`` (dict with
        projected_total / lower / upper / method) and ``charts``.
    """
    logger.info("Trend analysis started ...")
    cols = _resolve_columns(sales_df)
    df = sales_df.copy()

    date_col = cols["date"]
    total_col = cols["total"]

    if not date_col or not total_col:
        logger.warning("Missing required columns for trend analysis; returning empty result.")
        empty_df = pd.DataFrame()
        return {
            "daily_trend": empty_df,
            "weekly_trend": empty_df,
            "monthly_trend": empty_df,
            "yoy_comparison": empty_df,
            "moving_averages": empty_df,
            "growth_rates": empty_df,
            "projection": {"projected_total": 0, "lower": 0, "upper": 0, "method": "N/A", "horizon_days": 30, "seasonal_factor": 1.0, "upcoming_month": 1},
            "charts": {},
        }

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    daily = df.groupby(df[date_col].dt.normalize())[total_col].sum().sort_index()
    daily_trend = daily.reset_index()
    daily_trend.columns = ["Date", "Total"]

    iso = df[date_col].dt.isocalendar()
    weekly = df.groupby([iso["year"].astype(int), iso["week"].astype(int)])[total_col].sum()
    weekly_trend = weekly.reset_index()
    weekly_trend.columns = ["ISO_Year", "ISO_Week", "Total"]
    weekly_trend["Week"] = weekly_trend["ISO_Year"].astype(str) + "-W" + weekly_trend["ISO_Week"].astype(str).str.zfill(2)
    weekly_trend = weekly_trend[["Week", "Total"]]

    monthly_frame = pd.DataFrame({
        "Year": df[date_col].dt.year.astype(int),
        "Month": df[date_col].dt.month.astype(int),
        "Total": pd.to_numeric(df[total_col], errors="coerce"),
    })
    monthly = monthly_frame.groupby(["Year", "Month"])["Total"].sum().sort_index()
    monthly_trend = monthly.reset_index()
    monthly_trend.columns = ["Year", "Month", "Total"]
    monthly_trend["Year_Month"] = (
        monthly_trend["Year"].astype(str) + "-" + monthly_trend["Month"].astype(str).str.zfill(2)
    )
    monthly_trend = monthly_trend[["Year_Month", "Year", "Month", "Total"]]

    yoy_comparison = monthly_trend.pivot(index="Month", columns="Year", values="Total").fillna(0.0)

    moving_averages = daily_trend.copy()
    moving_averages["MA_7d"] = moving_averages["Total"].rolling(7, min_periods=1).mean().round(2)
    moving_averages["MA_30d"] = moving_averages["Total"].rolling(30, min_periods=1).mean().round(2)

    growth = monthly_trend.copy()
    growth["MoM_Pct"] = (growth["Total"].pct_change() * 100).round(2)
    growth["YoY_Pct"] = (growth["Total"].pct_change(12) * 100).round(2)
    growth_rates = growth[["Year_Month", "Total", "MoM_Pct", "YoY_Pct"]]

    # ---- simple seasonal-adjusted projection ----------------------------
    projection = _simple_projection(daily, df, date_col, total_col)

    # ---- charts ---------------------------------------------------------
    charts: Dict = {}
    ma_long = moving_averages.melt(id_vars="Date", value_vars=["Total", "MA_7d", "MA_30d"],
                                   var_name="Series", value_name="Value")
    daily_trend_chart = px.line(
        ma_long, x="Date", y="Value", color="Series", markers=False,
        color_discrete_sequence=PALETTE[:3],
        title="Daily Sales with 7-day and 30-day Moving Averages",
        labels={"Value": "Sales ($)"},
    )
    daily_trend_chart.update_layout(**_TEMPLATE)
    charts["daily_trend_chart"] = daily_trend_chart

    yoy_long = yoy_comparison.reset_index().melt(id_vars="Month", var_name="Year", value_name="Total")
    yoy_long["Year"] = yoy_long["Year"].astype(str)
    monthly_yoy_chart = px.line(
        yoy_long, x="Month", y="Total", color="Year", markers=True,
        color_discrete_sequence=PALETTE[:2],
        title="Monthly Revenue — Year over Year",
        labels={"Total": "Revenue ($)", "Month": "Month (1-12)"},
    )
    monthly_yoy_chart.update_layout(**_TEMPLATE, xaxis_dtick=1)
    charts["monthly_yoy_chart"] = monthly_yoy_chart

    monthly_pivot = df.pivot_table(
        index=df[date_col].dt.month, columns=df[date_col].dt.year,
        values=total_col, aggfunc="sum",
    ).fillna(0.0)
    monthly_pivot.columns = [str(c) for c in monthly_pivot.columns]
    seasonal_heatmap = px.imshow(
        monthly_pivot, text_auto=",.", aspect="auto",
        color_continuous_scale="Teal", origin="lower",
        title="Seasonal Revenue Heatmap (Month x Year)",
        labels={"x": "Year", "y": "Month", "color": "Revenue ($)"},
    )
    seasonal_heatmap.update_layout(**_TEMPLATE)
    charts["seasonal_heatmap"] = seasonal_heatmap

    logger.info(
        "Trend analysis complete: {} days, projection next 30d = ${:,.0f}.",
        len(daily_trend), projection["projected_total"],
    )
    return {
        "daily_trend": daily_trend,
        "weekly_trend": weekly_trend,
        "monthly_trend": monthly_trend,
        "yoy_comparison": yoy_comparison,
        "moving_averages": moving_averages,
        "growth_rates": growth_rates,
        "projection": projection,
        "charts": charts,
    }


def _simple_projection(daily: pd.Series, df: pd.DataFrame, date_col: str, total_col: str) -> Dict:
    """Seasonal-adjusted moving-average projection for the next 30 days.

    Method
    ------
    * ``seasonal factor`` for the upcoming month = (average daily revenue in
      that calendar month during the last full year) / (overall average
      daily revenue in that year).
    * ``projected_total`` = (mean daily revenue over the last 30 days) x
      factor x 30 days.
    * The confidence band scales the daily standard deviation by sqrt(30)
      to express it as a 30-day aggregate (±1 sigma).
    """
    daily = daily.sort_index()
    last_date = daily.index.max()
    last30 = daily.tail(30)
    daily_avg_30 = float(last30.mean())
    std_30 = float(last30.std(ddof=1)) if len(last30) > 1 else 0.0

    upcoming_month = 1 if last_date.month == 12 else last_date.month + 1
    reference_year = last_date.year if last_date.month < 12 else last_date.year  # last full year

    year_daily = daily[daily.index.year == reference_year]
    overall_avg = float(year_daily.mean()) if len(year_daily) else daily_avg_30 or 1.0
    month_avg = float(year_daily[year_daily.index.month == upcoming_month].mean()) if len(year_daily) else daily_avg_30
    seasonal_factor = month_avg / overall_avg if overall_avg else 1.0

    projected_total = daily_avg_30 * seasonal_factor * 30.0
    band = std_30 * np.sqrt(30.0)
    projection = {
        "projected_total": round(projected_total, 2),
        "lower": round(max(0.0, projected_total - band), 2),
        "upper": round(projected_total + band, 2),
        "method": "Seasonal-adjusted moving average",
        "horizon_days": 30,
        "seasonal_factor": round(float(seasonal_factor), 4),
        "upcoming_month": upcoming_month,
    }
    logger.info(
        "Projection: next 30d ${:,.0f} (factor {:.2f}, band +/- ${:,.0f}).",
        projected_total, seasonal_factor, band,
    )
    return projection


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_full_analytics(sales_df: pd.DataFrame) -> Dict:
    """Run RFM, product and trend analytics; persist a summary parquet.

    Returns
    -------
    dict
        ``rfm``, ``product``, ``trend`` (each the corresponding result
        dict/DataFrame) plus ``summary`` (tidy DataFrame written to
        ``data/processed/analytics_summary.parquet``).
    """
    logger.info("=== Full analytics started ===")
    try:
        rfm = rfm_analysis(sales_df)
        product = product_analysis(sales_df)
        trend = trend_analysis(sales_df)

        cols = _resolve_columns(sales_df)
        total_col = cols["total"]
        total_revenue = float(sales_df[total_col].sum()) if total_col and total_col in sales_df.columns else 0.0

        kpis = pd.DataFrame({
            "Section": "KPI",
            "Label": [
                "Proxy_Customers", "Total_Revenue", "Top_Product_Line",
                "Projected_Next_30d", "Projection_Lower", "Projection_Upper",
            ],
            "Value": [
                f"{len(rfm):,}",
                f"{total_revenue:,.2f}",
                product["best_sellers"].iloc[0]["Product_Line"] if not product["best_sellers"].empty else "N/A",
                f"{trend['projection']['projected_total']:,.2f}",
                f"{trend['projection']['lower']:,.2f}",
                f"{trend['projection']['upper']:,.2f}",
            ],
        })
        segment_counts = rfm["RFM_Segment"].value_counts().reset_index()
        segment_counts.columns = ["Label", "Value"]
        segment_counts["Section"] = "RFM_Segment"
        segment_counts["Value"] = segment_counts["Value"].astype(str)
        abc_summary = product["abc"][["Product_Line", "Revenue", "Revenue_Pct", "ABC_Class"]].copy()
        abc_summary = abc_summary.rename(columns={"Product_Line": "Label"})
        abc_summary["Section"] = "ABC_" + abc_summary["ABC_Class"]
        abc_summary["Value"] = abc_summary["Revenue"].map(lambda v: f"{v:,.2f}")
        recent_months = trend["monthly_trend"].tail(12).copy()
        recent_months["Label"] = recent_months["Year_Month"]
        recent_months["Value"] = recent_months["Total"].map(lambda v: f"{v:,.2f}")
        recent_months["Section"] = "Monthly_Revenue"

        summary = pd.concat(
            [
                kpis,
                segment_counts[["Section", "Label", "Value"]],
                abc_summary[["Section", "Label", "Value"]],
                recent_months[["Section", "Label", "Value"]],
            ],
            ignore_index=True,
        )
        out_path = config.PROCESSED_DATA_PATH / "analytics_summary.parquet"
        summary.to_parquet(out_path, index=False)
        logger.info("Analytics summary saved -> {} ({} rows).", out_path, len(summary))

        logger.info("=== Full analytics complete ===")
        return {"rfm": rfm, "product": product, "trend": trend, "summary": summary}
    except Exception:
        logger.exception("Full analytics failed")
        raise