"""Inventory analytics for the AI_Supermarket_BI project.

Pure pandas + numpy + plotly — no ML libraries. Provides:

* :func:`analyze_stock` — stock-status classification (stockouts, below
  reorder, overstock, adequate) with days-of-stock-left.
* :func:`calculate_safety_stock` — safety stock and reorder points from
  demand statistics at a configurable service level.
* :func:`generate_reorder_report` — action-labelled, urgency-sorted report.
* :func:`abc_analysis` — product-line ABC classification with a dual-axis
  plotly chart.
* :func:`run_inventory_pipeline` — chains everything and persists reports.

Demand model note: the sales table carries ``Product_Line`` while the
product catalogue carries ``Category`` — they share the same 6 values, so
demand statistics are computed per product line and inherited by every
product in that line.
"""
import math
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

PALETTE = px.colors.qualitative.Set2
PALETTE_HEX = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]

#: Z-scores for the supported service levels.
Z_SCORES = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}

#: Urgency ordering used by the reorder report.
ACTION_ORDER = ["🔴 URGENT REORDER", "🟡 REORDER SOON", "🟢 ADEQUATE", "🔵 OVERSTOCK"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _line_demand_stats(products_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-product-frame demand statistics joined from sales.

    Computes per product line: total quantity sold, average daily demand
    and the standard deviation of daily demand (on a complete-day basis),
    then left-joins them onto the product catalogue via Category.
    """
    sales = sales_df.copy()
    sales["Date"] = pd.to_datetime(sales["Date"])

    n_days = max(sales["Date"].dt.normalize().nunique(), 1)
    line_totals = sales.groupby("Product_Line", observed=True)["Quantity"].sum()

    daily_line = sales.groupby(
        [sales["Date"].dt.normalize(), "Product_Line"], observed=True
    )["Quantity"].sum()
    line_std = daily_line.groupby("Product_Line").std(ddof=1).fillna(0.0)

    stats = pd.DataFrame({
        "Product_Line": line_totals.index,
        "Line_Quantity": line_totals.values,
        "Avg_Daily_Demand": (line_totals / n_days).values,
        "Demand_Std": line_std.reindex(line_totals.index).fillna(0.0).values,
    })

    products = products_df.copy()
    merged = products.merge(stats, how="left", left_on="Category", right_on="Product_Line")
    merged["Avg_Daily_Demand"] = merged["Avg_Daily_Demand"].fillna(0.0)
    merged["Demand_Std"] = merged["Demand_Std"].fillna(0.0)
    return merged.drop(columns=["Product_Line"], errors="ignore")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_stock(products_df: pd.DataFrame, sales_df: pd.DataFrame) -> Dict:
    """Classify every product's stock situation against actual demand.

    Categories (mutually exclusive, in priority order):

    * ``stockouts``       — Stock == 0
    * ``below_reorder``   — 0 < Stock < Reorder_Level
    * ``overstock``       — Stock > 3 x Avg_Daily_Demand x 30 (> ~90 days)
    * ``adequate``        — everything else

    Returns
    -------
    dict
        ``{"summary": {category: count}, "details": DataFrame}`` where
        details carries Stock_Status and Days_Of_Stock_Left per product,
        sorted by Stock ascending (worst first).
    """
    logger.info("Stock analysis for {} products ...", len(products_df))
    merged = _line_demand_stats(products_df, sales_df)

    merged["Days_Of_Stock_Left"] = (
        merged["Stock"] / merged["Avg_Daily_Demand"].clip(lower=0.01)
    ).round(1)

    def _classify(row) -> str:
        if row["Stock"] == 0:
            return "stockout"
        if row["Stock"] < row["Reorder_Level"]:
            return "below_reorder"
        if row["Avg_Daily_Demand"] > 0 and row["Stock"] > 3 * row["Avg_Daily_Demand"] * 30:
            return "overstock"
        return "adequate"

    merged["Stock_Status"] = merged.apply(_classify, axis=1)

    summary = {
        "stockouts": int((merged["Stock_Status"] == "stockout").sum()),
        "below_reorder": int((merged["Stock_Status"] == "below_reorder").sum()),
        "overstock": int((merged["Stock_Status"] == "overstock").sum()),
        "adequate": int((merged["Stock_Status"] == "adequate").sum()),
        "total_products": int(len(merged)),
    }
    details = merged[[
        "Product_ID", "Product_Name", "Category", "Stock", "Reorder_Level",
        "Avg_Daily_Demand", "Days_Of_Stock_Left", "Stock_Status",
    ]].sort_values("Stock").reset_index(drop=True)

    logger.info("Stock summary: {}", summary)
    return {"summary": summary, "details": details}


def calculate_safety_stock(
    products_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    service_level: float = 0.95,
    lead_time: int = 7,
) -> pd.DataFrame:
    """Compute safety stock and reorder points per product.

    ``Safety_Stock = Z x Demand_Std x sqrt(lead_time)`` (rounded up) and
    ``Reorder_Point = ceil(Avg_Daily_Demand x lead_time + Safety_Stock)``.

    Parameters
    ----------
    products_df:
        Product catalogue (Stock column is the current stock).
    sales_df:
        Sales history used for the demand statistics.
    service_level:
        One of 0.90 / 0.95 / 0.99 (defaults to 0.95 -> Z = 1.65).
    lead_time:
        Supplier lead time in days (default 7).

    Returns
    -------
    pandas.DataFrame
        Product_ID, Product_Name, Current_Stock, Avg_Daily_Demand,
        Demand_Std, Safety_Stock, Reorder_Point.
    """
    z = Z_SCORES.get(service_level, 1.65)
    logger.info(
        "Safety stock calc: {} products, service level {} (Z={}), lead {} days.",
        len(products_df), service_level, z, lead_time,
    )
    merged = _line_demand_stats(products_df, sales_df)

    merged["Safety_Stock"] = np.ceil(
        z * merged["Demand_Std"] * math.sqrt(lead_time)
    ).astype(int)
    merged["Reorder_Point"] = np.ceil(
        merged["Avg_Daily_Demand"] * lead_time + merged["Safety_Stock"]
    ).astype(int)

    result = merged[[
        "Product_ID", "Product_Name", "Stock", "Avg_Daily_Demand",
        "Demand_Std", "Safety_Stock", "Reorder_Point",
    ]].rename(columns={"Stock": "Current_Stock"}).reset_index(drop=True)
    logger.info(
        "Safety stock complete: median reorder point = {}, {} products below it.",
        int(result["Reorder_Point"].median()),
        int((result["Current_Stock"] < result["Reorder_Point"]).sum()),
    )
    return result


def generate_reorder_report(safety_stock_df: pd.DataFrame) -> pd.DataFrame:
    """Turn safety-stock output into an urgency-sorted action report.

    Actions:

    * Stock == 0                -> "🔴 URGENT REORDER"
    * Stock < Reorder_Point     -> "🟡 REORDER SOON"
    * Stock < 2xReorder_Point   -> "🟢 ADEQUATE"
    * otherwise                 -> "🔵 OVERSTOCK"

    Returns
    -------
    pandas.DataFrame
        The input columns plus ``Action`` and ``Headroom`` (units above /
        below the reorder point), sorted URGENT -> REORDER -> ADEQUATE ->
        OVERSTOCK.
    """
    report = safety_stock_df.copy()

    def _action(row) -> str:
        if row["Current_Stock"] == 0:
            return ACTION_ORDER[0]
        if row["Current_Stock"] < row["Reorder_Point"]:
            return ACTION_ORDER[1]
        if row["Current_Stock"] < row["Reorder_Point"] * 2:
            return ACTION_ORDER[2]
        return ACTION_ORDER[3]

    report["Action"] = report.apply(_action, axis=1)
    report["Headroom"] = (report["Current_Stock"] - report["Reorder_Point"]).astype(int)
    report["Action"] = pd.Categorical(report["Action"], categories=ACTION_ORDER, ordered=True)
    report = report.sort_values(["Action", "Headroom"]).reset_index(drop=True)
    report["Action"] = report["Action"].astype(str)

    counts = report["Action"].value_counts().to_dict()
    logger.info("Reorder report: {} rows; actions: {}", len(report), counts)
    return report


def abc_analysis(sales_df: pd.DataFrame) -> Dict:
    """ABC-classify product lines by revenue share (A<=80%, B<=95%, C>95%).

    Returns
    -------
    dict
        ``{"abc_df": DataFrame, "plot": plotly figure}`` — the figure shows
        revenue bars with a cumulative-share line on a secondary axis.
    """
    revenue = (
        sales_df.groupby("Product_Line", observed=True)["Total"].sum()
        .sort_values(ascending=False).reset_index()
    )
    revenue["Revenue_Pct"] = (revenue["Total"] / revenue["Total"].sum() * 100).round(2)
    revenue["Cumulative_Pct"] = revenue["Revenue_Pct"].cumsum().round(2)
    revenue["ABC_Class"] = np.where(
        revenue["Cumulative_Pct"] <= 80.0, "A",
        np.where(revenue["Cumulative_Pct"] <= 95.0, "B", "C"),
    )
    revenue = revenue.rename(columns={"Total": "Revenue"})

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=revenue["Product_Line"], y=revenue["Revenue"], name="Revenue",
               marker_color=PALETTE_HEX[0]),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=revenue["Product_Line"], y=revenue["Cumulative_Pct"],
                   name="Cumulative %", mode="lines+markers",
                   line=dict(color=PALETTE_HEX[1], dash="dash")),
        secondary_y=True,
    )
    fig.update_layout(
        title="ABC Analysis — Product Lines by Revenue",
        template=config.PLOTLY_TEMPLATE, height=config.CHART_HEIGHT,
    )
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
    fig.update_xaxes(tickangle=-20)

    logger.info(
        "ABC: {} A-class, {} B-class, {} C-class lines.",
        int((revenue["ABC_Class"] == "A").sum()),
        int((revenue["ABC_Class"] == "B").sum()),
        int((revenue["ABC_Class"] == "C").sum()),
    )
    return {"abc_df": revenue, "plot": fig}


def run_inventory_pipeline(products_df: pd.DataFrame, sales_df: pd.DataFrame) -> Dict:
    """Chain stock analysis, safety stock, reorder report and ABC analysis.

    Also persists the reorder report and ABC table to the processed-data
    directory (``reorder_report.csv`` / ``abc_analysis.csv``).

    Returns
    -------
    dict
        ``stock`` (summary+details), ``safety_stock``, ``reorder_report``,
        ``abc`` (abc_df+plot).
    """
    logger.info("=== Inventory pipeline started ===")
    try:
        stock = analyze_stock(products_df, sales_df)
        safety = calculate_safety_stock(
            products_df, sales_df,
            service_level=0.95,  # matches config.SERVICE_LEVEL_Z = 1.65
            lead_time=config.LEAD_TIME_DAYS,
        )
        reorder = generate_reorder_report(safety)
        abc = abc_analysis(sales_df)

        try:
            config.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
            reorder.to_csv(config.REORDER_REPORT_FILE, index=False)
            abc["abc_df"].to_csv(config.ABC_ANALYSIS_FILE, index=False)
            logger.info("Saved {} and {}.", config.REORDER_REPORT_FILE.name, config.ABC_ANALYSIS_FILE.name)
        except Exception as exc:
            logger.warning("Could not persist inventory CSVs: {}", exc)

        logger.info("=== Inventory pipeline complete ===")
        return {
            "stock": stock,
            "safety_stock": safety,
            "reorder_report": reorder,
            "abc": abc,
        }
    except Exception:
        logger.exception("Inventory pipeline failed")
        raise
