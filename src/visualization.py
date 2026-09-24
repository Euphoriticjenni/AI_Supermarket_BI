"""Chart factory for the AI_Supermarket_BI project.

Every function takes a (processed) sales DataFrame and returns a ready-to-
render plotly figure, used by the Streamlit dashboard, the EDA notebook and
the PDF report generator.

Conventions
-----------
* One shared qualitative palette (``px.colors.qualitative.Set2``) so all
  charts look consistent across pages.
* Every chart has a title, axis labels and hover data.
* Figures use ``config.PLOTLY_TEMPLATE`` ("plotly_white") and a standard
  height; widths stay responsive for Streamlit embedding.
* Aggregations are done here (groupby) — the dashboard passes full data and
  lets these helpers reduce it, keeping page code thin.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

#: Shared qualitative palette (consistent across every chart in the project).
PALETTE = px.colors.qualitative.Set2

#: The same Set2 colors as hex strings — matplotlib rejects plotly's
#: "rgb(r,g,b)" format, so the fallback renderer uses these instead.
PALETTE_HEX = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]

_TEMPLATE = {"template": config.PLOTLY_TEMPLATE, "height": config.CHART_HEIGHT}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _ensure_chart_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal/financial columns on the fly when missing.

    Works with either processed data (``sales_processed.parquet``) or the
    feature-enriched frame (``sales_features.parquet``): any of ``Year``,
    ``Month``, ``Month_Name``, ``Purchase_Hour``, ``Day_Name`` and
    ``Profit`` absent from the input are recomputed from ``Date``/``Time``/
    ``COGS`` so every chart function accepts both frames unchanged.
    """
    df = df.copy()
    if "Date" not in df.columns:
        raise KeyError("Sales DataFrame must contain a 'Date' column")
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "Purchase_Hour" not in df.columns and "Time" in df.columns:
        df["Purchase_Hour"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour
    if "Day_Name" not in df.columns:
        df["Day_Name"] = df["Date"].dt.day_name()
    if "Month" not in df.columns:
        df["Month"] = df["Date"].dt.month
    if "Month_Name" not in df.columns:
        df["Month_Name"] = df["Date"].dt.month_name()
    if "Year" not in df.columns:
        df["Year"] = df["Date"].dt.year
    if "Profit" not in df.columns and "COGS" in df.columns:
        df["Profit"] = df["Total"] - df["COGS"]
    return df


def _daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Total revenue per calendar day."""
    daily = (
        df.groupby(df["Date"].dt.normalize().rename("Date"), observed=True)["Total"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )
    return daily


def _monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Total revenue per (Year, Month) with ordered month names."""
    monthly = (
        df.groupby(["Year", "Month", "Month_Name"], observed=True)["Total"]
        .sum()
        .reset_index()
        .sort_values(["Year", "Month"])
    )
    monthly["Year_Month"] = (
        monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)
    )
    return monthly


# ---------------------------------------------------------------------------
# 1-3: Sales trends
# ---------------------------------------------------------------------------
def plot_daily_sales(df: pd.DataFrame) -> go.Figure:
    """Line chart of daily total sales with a 7-day moving-average overlay."""
    try:
        df = _ensure_chart_columns(df)
        daily = _daily_totals(df)
        daily["MA_7d"] = daily["Total"].rolling(7, min_periods=1).mean()
        fig = px.line(
            daily, x="Date", y="Total",
            title="Daily Sales Trend",
            labels={"Total": "Sales ($)", "Date": "Date"},
            template=config.PLOTLY_TEMPLATE,
        )
        fig.add_scatter(
            x=daily["Date"], y=daily["MA_7d"], mode="lines",
            name="7-Day Moving Average", line=dict(color=PALETTE[1], width=3),
        )
        fig.update_traces(line=dict(color=PALETTE[0], width=1.5), selector=dict(name="Total"))
        fig.update_layout(**_TEMPLATE)
        return fig
    except Exception:
        logger.exception("plot_daily_sales failed")
        raise


def plot_monthly_sales(df: pd.DataFrame) -> go.Figure:
    """Bar chart of monthly revenue, coloured by year."""
    try:
        df = _ensure_chart_columns(df)
        monthly = _monthly_totals(df)
        fig = px.bar(
            monthly, x="Year_Month", y="Total", color="Year",
            title="Monthly Revenue",
            labels={"Total": "Revenue ($)", "Year_Month": "Month"},
            color_discrete_sequence=PALETTE,
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_layout(**_TEMPLATE, xaxis_tickangle=-45)
        return fig
    except Exception:
        logger.exception("plot_monthly_sales failed")
        raise


def plot_revenue_trend(df: pd.DataFrame) -> go.Figure:
    """Daily revenue with a manual OLS linear trend line as a second trace."""
    try:
        df = _ensure_chart_columns(df)
        daily = _daily_totals(df)
        x_num = (daily["Date"] - daily["Date"].min()).dt.days.to_numpy(dtype=float)
        y = daily["Total"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_num, y, 1)  # manual OLS fit
        trend = slope * x_num + intercept
        r_squared = 1 - np.sum((y - trend) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)

        fig = px.line(
            daily, x="Date", y="Total",
            title=f"Revenue Trend (linear fit: {slope:+,.0f} $/day, R² = {r_squared:.3f})",
            labels={"Total": "Revenue ($)", "Date": "Date"},
            template=config.PLOTLY_TEMPLATE,
        )
        fig.data[0].line.color = PALETTE[0]
        fig.add_scatter(
            x=daily["Date"], y=trend, mode="lines", name="OLS Trend",
            line=dict(color=PALETTE[1], width=3, dash="dash"),
        )
        fig.update_layout(**_TEMPLATE)
        return fig
    except Exception:
        logger.exception("plot_revenue_trend failed")
        raise


# ---------------------------------------------------------------------------
# 4-6: Dimensions
# ---------------------------------------------------------------------------
def plot_sales_by_branch(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of revenue by branch, split by product line."""
    try:
        df = _ensure_chart_columns(df)
        agg = (
            df.groupby(["Branch", "Product_Line"], observed=True)["Total"]
            .sum().reset_index()
        )
        fig = px.bar(
            agg, x="Branch", y="Total", color="Product_Line", barmode="group",
            title="Sales by Branch and Product Line",
            labels={"Total": "Revenue ($)", "Branch": "Branch"},
            color_discrete_sequence=PALETTE,
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_layout(**_TEMPLATE)
        return fig
    except Exception:
        logger.exception("plot_sales_by_branch failed")
        raise


def plot_sales_by_city(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of revenue by city."""
    try:
        df = _ensure_chart_columns(df)
        agg = df.groupby("City", observed=True)["Total"].sum().reset_index()
        agg = agg.sort_values("Total")
        fig = px.bar(
            agg, y="City", x="Total", orientation="h", color="City",
            title="Revenue by City",
            labels={"Total": "Revenue ($)", "City": "City"},
            color_discrete_sequence=PALETTE,
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_layout(**_TEMPLATE, showlegend=False)
        return fig
    except Exception:
        logger.exception("plot_sales_by_city failed")
        raise


def plot_sales_by_product_line(df: pd.DataFrame) -> go.Figure:
    """Treemap of revenue share by product line."""
    try:
        df = _ensure_chart_columns(df)
        agg = df.groupby("Product_Line", observed=True)["Total"].sum().reset_index()
        fig = px.treemap(
            agg, path=["Product_Line"], values="Total", color="Product_Line",
            title="Revenue Share by Product Line",
            color_discrete_sequence=PALETTE,
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_traces(textinfo="label+percent entry+value")
        fig.update_layout(**_TEMPLATE)
        return fig
    except Exception:
        logger.exception("plot_sales_by_product_line failed")
        raise


# ---------------------------------------------------------------------------
# 7-9: Customer & payment mix
# ---------------------------------------------------------------------------
def plot_customer_type_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart of transactions by customer type."""
    try:
        df = _ensure_chart_columns(df)
        counts = df["Customer_Type"].value_counts().reset_index()
        counts.columns = ["Customer_Type", "Transactions"]
        fig = px.pie(
            counts, names="Customer_Type", values="Transactions",
            title="Customer Type Distribution (transactions)",
            color_discrete_sequence=PALETTE, hole=0,
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:,} (%{percent})")
        fig.update_layout(**_TEMPLATE)
        return fig
    except Exception:
        logger.exception("plot_customer_type_distribution failed")
        raise


def plot_gender_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart (hole=0.4) of transactions by gender."""
    try:
        df = _ensure_chart_columns(df)
        counts = df["Gender"].value_counts().reset_index()
        counts.columns = ["Gender", "Transactions"]
        fig = px.pie(
            counts, names="Gender", values="Transactions",
            title="Gender Distribution (transactions)",
            color_discrete_sequence=PALETTE, hole=0.4,
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:,} (%{percent})")
        fig.update_layout(**_TEMPLATE)
        return fig
    except Exception:
        logger.exception("plot_gender_distribution failed")
        raise


def plot_payment_methods(df: pd.DataFrame) -> go.Figure:
    """Bar chart of transaction counts per payment method."""
    try:
        df = _ensure_chart_columns(df)
        counts = df["Payment"].value_counts().reset_index()
        counts.columns = ["Payment", "Transactions"]
        fig = px.bar(
            counts, x="Payment", y="Transactions", color="Payment",
            title="Payment Method Breakdown",
            labels={"Transactions": "Transactions", "Payment": "Payment Method"},
            color_discrete_sequence=PALETTE,
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_layout(**_TEMPLATE, showlegend=False)
        return fig
    except Exception:
        logger.exception("plot_payment_methods failed")
        raise


# ---------------------------------------------------------------------------
# 10-11: Temporal patterns
# ---------------------------------------------------------------------------
def plot_hourly_sales_pattern(df: pd.DataFrame) -> go.Figure:
    """Line chart of average transaction value by hour of day."""
    try:
        df = _ensure_chart_columns(df)
        agg = df.groupby("Purchase_Hour", observed=True)["Total"].mean().reset_index()
        agg.columns = ["Hour", "Avg_Transaction"]
        fig = px.line(
            agg, x="Hour", y="Avg_Transaction", markers=True,
            title="Average Transaction Value by Hour of Day",
            labels={"Avg_Transaction": "Avg Transaction ($)", "Hour": "Hour of Day"},
            template=config.PLOTLY_TEMPLATE,
        )
        fig.data[0].line.color = PALETTE[0]
        fig.update_xaxes(dtick=1)
        fig.update_layout(**_TEMPLATE)
        return fig
    except Exception:
        logger.exception("plot_hourly_sales_pattern failed")
        raise


def plot_day_of_week_pattern(df: pd.DataFrame) -> go.Figure:
    """Bar chart of total revenue by weekday, ordered Monday -> Sunday."""
    try:
        df = _ensure_chart_columns(df)
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        agg = (
            df.groupby("Day_Name", observed=True)["Total"].sum()
            .reindex(order).reset_index()
        )
        agg.columns = ["Day_Name", "Total"]
        fig = px.bar(
            agg, x="Day_Name", y="Total", color="Day_Name",
            category_orders={"Day_Name": order},
            title="Revenue by Day of Week",
            labels={"Total": "Revenue ($)", "Day_Name": "Day"},
            color_discrete_sequence=PALETTE,
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_layout(**_TEMPLATE, showlegend=False)
        return fig
    except Exception:
        logger.exception("plot_day_of_week_pattern failed")
        raise


# ---------------------------------------------------------------------------
# 12: Correlation heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of correlations between the numeric business columns."""
    try:
        df = _ensure_chart_columns(df)
        numeric_cols = [
            "Unit_Price", "Quantity", "Total", "COGS", "Gross_Income",
            "Rating", "Profit", "Purchase_Hour",
        ]
        available = [c for c in numeric_cols if c in df.columns]
        corr = df[available].corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlation Matrix (numeric features)",
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_layout(**_TEMPLATE)
        return fig
    except Exception:
        logger.exception("plot_correlation_heatmap failed")
        raise


# ---------------------------------------------------------------------------
# 13: Seasonal decomposition
# ---------------------------------------------------------------------------
def plot_seasonal_decomposition(df: pd.DataFrame, period: int = 30) -> go.Figure:
    """Four-panel seasonal decomposition of daily sales (period=30 days).

    Panels: observed, trend, seasonal and residuals, computed with
    statsmodels' ``seasonal_decompose`` on the daily revenue series.
    """
    try:
        df = _ensure_chart_columns(df)
        from statsmodels.tsa.seasonal import seasonal_decompose

        daily = _daily_totals(df).set_index("Date")["Total"]
        # Daily grids can have gaps; fill with the mean before decomposing.
        full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        series = daily.reindex(full_index).interpolate().fillna(daily.mean())
        result = seasonal_decompose(series, model="additive", period=period)

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        )
        panels = [
            (result.observed, PALETTE[0]),
            (result.trend, PALETTE[1]),
            (result.seasonal, PALETTE[2]),
            (result.resid, PALETTE[3]),
        ]
        for row, (panel, color) in enumerate(panels, start=1):
            fig.add_trace(
                go.Scatter(x=panel.index, y=panel.values, mode="lines",
                           name=panel.name or f"panel{row}",
                           line=dict(color=color, width=1.4),
                           hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra></extra>"),
                row=row, col=1,
            )
        fig.update_layout(
            title=f"Seasonal Decomposition of Daily Sales (period={period} days)",
            template=config.PLOTLY_TEMPLATE, height=900,
            showlegend=False,
        )
        fig.update_yaxes(title_text="Sales ($)", row=1, col=1)
        return fig
    except Exception:
        logger.exception("plot_seasonal_decomposition failed")
        raise


# ---------------------------------------------------------------------------
# 14-16: Ratings, profit, YoY
# ---------------------------------------------------------------------------
def plot_rating_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of ratings (20 bins) overlaid per branch."""
    try:
        df = _ensure_chart_columns(df)
        fig = px.histogram(
            df, x="Rating", nbins=20, color="Branch", barmode="overlay",
            title="Customer Rating Distribution by Branch",
            labels={"Rating": "Rating", "count": "Transactions"},
            color_discrete_sequence=PALETTE, opacity=0.65,
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_layout(**_TEMPLATE)
        return fig
    except Exception:
        logger.exception("plot_rating_distribution failed")
        raise


def plot_profit_by_category(df: pd.DataFrame) -> go.Figure:
    """Bar chart of total profit by product line, sorted descending."""
    try:
        df = _ensure_chart_columns(df)
        agg = (
            df.groupby("Product_Line", observed=True)["Profit"].sum()
            .sort_values(ascending=False).reset_index()
        )
        fig = px.bar(
            agg, x="Product_Line", y="Profit", color="Product_Line",
            title="Total Profit by Product Line",
            labels={"Profit": "Profit ($)", "Product_Line": "Product Line"},
            color_discrete_sequence=PALETTE,
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_layout(**_TEMPLATE, showlegend=False, xaxis_tickangle=-20)
        return fig
    except Exception:
        logger.exception("plot_profit_by_category failed")
        raise


def plot_yoy_comparison(df: pd.DataFrame) -> go.Figure:
    """Overlaid line chart comparing monthly revenue across years."""
    try:
        df = _ensure_chart_columns(df)
        monthly = _monthly_totals(df)
        fig = px.line(
            monthly, x="Month_Name", y="Total", color="Year", markers=True,
            category_orders={
                "Month_Name": ["January", "February", "March", "April", "May", "June",
                               "July", "August", "September", "October", "November", "December"],
            },
            title="Year-over-Year Monthly Revenue Comparison",
            labels={"Total": "Revenue ($)", "Month_Name": "Month"},
            color_discrete_sequence=PALETTE,
            template=config.PLOTLY_TEMPLATE,
        )
        fig.update_layout(**_TEMPLATE)
        return fig
    except Exception:
        logger.exception("plot_yoy_comparison failed")
        raise


# ---------------------------------------------------------------------------
# KPI helper
# ---------------------------------------------------------------------------
def get_kpi_summary(df: pd.DataFrame) -> Dict:
    """Compute the headline KPIs shown on the dashboard home page.

    Returns
    -------
    dict
        ``total_revenue``, ``total_profit``, ``total_orders`` (unique
        invoices), ``avg_order_value``, ``unique_customers`` (from the
        customer base file when readable, else ``None``), ``avg_rating``,
        ``best_branch``, ``best_product_line``.
    """
    try:
        df = _ensure_chart_columns(df)
        total_orders = df["Invoice_ID"].nunique()
        total_revenue = float(df["Total"].sum())
        summary: Dict = {
            "total_revenue": total_revenue,
            "total_profit": float(df["Profit"].sum()) if "Profit" in df.columns else total_revenue * config.GROSS_MARGIN_PCT / 100,
            "total_orders": total_orders,
            "avg_order_value": total_revenue / max(total_orders, 1),
            "unique_customers": None,
            "avg_rating": float(df["Rating"].mean()),
            "best_branch": str(df.groupby("Branch", observed=True)["Total"].sum().idxmax()),
            "best_product_line": str(df.groupby("Product_Line", observed=True)["Total"].sum().idxmax()),
        }
        # The sales table has no customer key; report the size of the
        # generated customer base when the file is available.
        try:
            if config.CUSTOMERS_DATA_FILE.exists():
                summary["unique_customers"] = int(pd.read_csv(config.CUSTOMERS_DATA_FILE).shape[0])
        except Exception:
            logger.warning("Could not read customer base for unique_customers KPI.")

        logger.info(
            "KPI summary: revenue ${:,.0f}, profit ${:,.0f}, {:,} orders, AOV ${:,.2f}.",
            summary["total_revenue"], summary["total_profit"],
            summary["total_orders"], summary["avg_order_value"],
        )
        return summary
    except Exception:
        logger.exception("get_kpi_summary failed")
        raise


# ---------------------------------------------------------------------------
# Static image export (matplotlib only — kaleido is banned project-wide)
# ---------------------------------------------------------------------------
# NOTE: kaleido / plotly .to_image() is deliberately NOT used anywhere in
# this project: its bundled Chromium hangs on some Windows hosts. All
# static PNG export goes through matplotlib (Agg backend) instead.


def render_trend_image_matplotlib(df: pd.DataFrame, width: int = 900, height: int = 420) -> Optional[bytes]:
    """Render the daily-sales trend chart with matplotlib (no kaleido needed).

    Fallback used by the PDF report when kaleido is unavailable. Uses the
    Agg backend, so it works fully headless.

    Returns
    -------
    bytes or None
        PNG bytes, or ``None`` when rendering failed.
    """
    try:
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        daily = _daily_totals(df)
        dpi = 100
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax.plot(daily["Date"], daily["Total"], color=PALETTE_HEX[0], linewidth=0.9)
        ax.set_title("Daily Sales Trend", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales ($)")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        png = buffer.getvalue()
        logger.info("Rendered matplotlib fallback trend chart ({} KB).", len(png) // 1024)
        return png
    except Exception:
        logger.exception("matplotlib fallback rendering failed")
        return None
