"""Automated executive PDF reports for the AI_Supermarket_BI project.

Builds a multi-page executive report with fpdf2. All charts are rendered
with matplotlib to PNG and embedded — plotly/kaleido are NEVER used for
image export (kaleido is broken on this host).

Note: the PDF core font Helvetica is latin-1 only, so every string written
to the document is plain ASCII (no emoji, en-dashes or currency symbols
beyond '$').
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # headless — must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from sqlalchemy import create_engine

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

#: matplotlib-safe mirror of the project Set2 palette.
PALETTE_HEX = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]

#: Column widths for the two-column KPI table (must sum <= 190 mm).
_KPI_COL_WIDTHS = (110, 80)


# ---------------------------------------------------------------------------
# Chart rendering (matplotlib only)
# ---------------------------------------------------------------------------
def render_chart_matplotlib(
    data_dict: Dict[str, float],
    chart_type: str,
    filepath: Path,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> Path:
    """Render a simple chart from a {label: value} dict and save as PNG.

    Parameters
    ----------
    data_dict:
        Mapping of category label -> numeric value.
    chart_type:
        One of ``"bar"``, ``"line"`` or ``"pie"``.
    filepath:
        Destination PNG path (parents are created).
    title, xlabel, ylabel:
        Optional chart decorations.

    Returns
    -------
    pathlib.Path
        The path written (also returned when rendering fails so callers
        can skip embedding gracefully — check ``path.exists()``).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    keys = [str(k) for k in data_dict.keys()]
    values = [float(v) for v in data_dict.values()]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    try:
        if chart_type == "bar":
            ax.bar(keys, values, color=PALETTE_HEX[: len(keys) % 6 or 1] * 6)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=25, ha="right", fontsize=8)
            if ylabel:
                ax.set_ylabel(ylabel)
        elif chart_type == "line":
            ax.plot(keys, values, color=PALETTE_HEX[0], marker="o", linewidth=1.8)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
            ax.grid(True, alpha=0.3)
            if ylabel:
                ax.set_ylabel(ylabel)
        elif chart_type == "pie":
            if len(values) > 6:
                keep = slice(0, 5)
                keys, values = list(keys)[keep], list(values)[keep]
            ax.pie(
                values, labels=keys, autopct="%1.1f%%", startangle=90,
                colors=PALETTE_HEX * 2,
            )
        else:
            raise ValueError(f"Unsupported chart_type: {chart_type!r}")
        if title:
            ax.set_title(title, fontsize=11)
        if xlabel and chart_type != "pie":
            ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.savefig(filepath, format="png")
    except Exception:
        logger.exception("matplotlib chart rendering failed for {}", filepath.name)
        raise
    finally:
        plt.close(fig)  # always free the figure
    logger.info("Rendered chart -> {} ({}).", filepath.name, chart_type)
    return filepath


# ---------------------------------------------------------------------------
# PDF building blocks
# ---------------------------------------------------------------------------
class _ReportPDF(FPDF):
    """FPDF subclass with a title page helper and a simple table renderer."""

    def heading(self, text: str) -> None:
        """Write a bold section heading with spacing."""
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def paragraph(self, text: str) -> None:
        """Write a wrapped body paragraph."""
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def table(self, headers, rows, widths) -> None:
        """Render a bordered table: a bold header row plus data rows.

        Parameters
        ----------
        headers:
            Column captions.
        rows:
            Iterable of row tuples (all values are str()-ed).
        widths:
            Column widths in mm (sum <= 190).
        """
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(235, 235, 235)
        for width, header in zip(widths, headers):
            self.cell(width, 7, str(header), border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9)
        for row in rows:
            for width, value in zip(widths, row):
                self.cell(width, 6, str(value), border=1)
            self.ln()
        self.ln(2)


def _load_from_database() -> Dict[str, pd.DataFrame]:
    """Load the three tables from the SQLite database via SQLAlchemy."""
    engine = create_engine(f"sqlite:///{Path(config.DATABASE_PATH).as_posix()}")
    try:
        frames = {
            "sales": pd.read_sql_table("sales", engine),
            "products": pd.read_sql_table("products", engine),
            "customers": pd.read_sql_table("customers", engine),
        }
    finally:
        engine.dispose()
    logger.info(
        "Loaded from database: {} sales, {} products, {} customers.",
        len(frames["sales"]), len(frames["products"]), len(frames["customers"]),
    )
    return frames


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_executive_report(
    sales_df: Optional[pd.DataFrame] = None,
    products_df: Optional[pd.DataFrame] = None,
    customers_df: Optional[pd.DataFrame] = None,
) -> str:
    """Generate the 7-page executive PDF report.

    Pages: 1 Title, 2 KPI Summary, 3 Top Products, 4 Branch Performance,
    5 Customer Overview, 6 Monthly Trends, 7 Recommendations.

    Parameters
    ----------
    sales_df, products_df, customers_df:
        Input frames; any left as ``None`` is loaded from the database.

    Returns
    -------
    str
        Absolute path of the written PDF.
    """
    logger.info("=== Executive report generation started ===")
    try:
        if sales_df is None or products_df is None or customers_df is None:
            frames = _load_from_database()
            sales_df = sales_df if sales_df is not None else frames["sales"]
            products_df = products_df if products_df is not None else frames["products"]
            customers_df = customers_df if customers_df is not None else frames["customers"]

        sales = sales_df.copy()
        sales["Date"] = pd.to_datetime(sales["Date"])

        # ---- headline KPIs ------------------------------------------------
        total_revenue = float(sales["Total"].sum())
        total_profit = float(sales["Gross_Income"].sum())
        total_orders = int(sales["Invoice_ID"].nunique())
        avg_order_value = total_revenue / max(total_orders, 1)
        avg_rating = float(sales["Rating"].mean())

        line_revenue = (
            sales.groupby("Product_Line", observed=True)["Total"].sum()
            .sort_values(ascending=False)
        )
        best_product = str(line_revenue.index[0])
        worst_product = str(line_revenue.index[-1])

        branch_stats = sales.groupby("Branch", observed=True).agg(
            Revenue=("Total", "sum"), Profit=("Gross_Income", "sum"),
        ).sort_values("Revenue", ascending=False)
        best_branch = str(branch_stats.index[0])

        day_key = sales["Date"].dt.normalize()
        is_weekend = sales["Date"].dt.dayofweek >= 5
        n_weekend_days = max(sales.loc[is_weekend].assign(_d=day_key)["_d"].nunique(), 1)
        n_weekday_days = max(sales.loc[~is_weekend].assign(_d=day_key)["_d"].nunique(), 1)
        weekend_daily = sales.loc[is_weekend, "Total"].sum() / n_weekend_days
        weekday_daily = sales.loc[~is_weekend, "Total"].sum() / n_weekday_days
        weekend_uplift = (weekend_daily / weekday_daily - 1) * 100 if weekday_daily else 0.0

        stockouts = int((products_df["Stock"] == 0).sum())

        config.REPORT_PATH.mkdir(parents=True, exist_ok=True)
        chart_dir = config.REPORT_PATH / "_charts"
        chart_paths = {}

        pdf = _ReportPDF(format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", "", 10)

        # ---------------- Page 1: title ------------------------------------
        pdf.add_page()
        pdf.ln(50)
        pdf.set_font("Helvetica", "B", 22)
        pdf.cell(0, 12, "Supermarket Business Intelligence Report", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Period: {sales['Date'].min():%Y-%m-%d} to {sales['Date'].max():%Y-%m-%d}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        pdf.set_draw_color(100, 100, 100)
        pdf.line(30, pdf.get_y(), 180, pdf.get_y())

        # ---------------- Page 2: KPI summary ------------------------------
        pdf.add_page()
        pdf.heading("KPI Summary")
        kpi_rows = [
            ("Total Revenue", f"${total_revenue:,.2f}"),
            ("Total Profit", f"${total_profit:,.2f}"),
            ("Total Orders", f"{total_orders:,}"),
            ("Avg Order Value", f"${avg_order_value:,.2f}"),
            ("Unique Products", f"{len(products_df):,}"),
            ("Customer Base", f"{len(customers_df):,}"),
            ("Avg Rating", f"{avg_rating:.2f} / 10"),
            ("Best Branch", f"Branch {best_branch}"),
            ("Best Product Line", best_product),
        ]
        pdf.table(["Metric", "Value"], kpi_rows, _KPI_COL_WIDTHS)

        # ---------------- Page 3: top products ------------------------------
        pdf.add_page()
        pdf.heading("Top Products")
        top_lines = line_revenue.head(10)
        chart_paths["top_products"] = render_chart_matplotlib(
            {k: float(v) for k, v in top_lines.items()}, "bar",
            chart_dir / "top_products.png",
            title="Revenue by Product Line", ylabel="Revenue ($)",
        )
        pdf.image(str(chart_paths["top_products"]), x=15, w=180)
        pdf.ln(4)
        pdf.table(
            ["Product Line", "Revenue ($)", "Share (%)"],
            [
                (name, f"${value:,.2f}", f"{value / total_revenue * 100:.1f}")
                for name, value in top_lines.items()
            ],
            (90, 55, 45),
        )

        # ---------------- Page 4: branch performance ------------------------
        pdf.add_page()
        pdf.heading("Branch Performance")
        chart_paths["branches"] = render_chart_matplotlib(
            {f"Branch {b}": float(r.Revenue) for b, r in branch_stats.iterrows()}, "bar",
            chart_dir / "branches.png",
            title="Revenue by Branch", ylabel="Revenue ($)",
        )
        pdf.image(str(chart_paths["branches"]), x=15, w=180)
        pdf.ln(4)
        pdf.table(
            ["Branch", "Revenue ($)", "Profit ($)"],
            [
                (str(branch), f"${row.Revenue:,.2f}", f"${row.Profit:,.2f}")
                for branch, row in branch_stats.iterrows()
            ],
            (60, 65, 65),
        )

        # ---------------- Page 5: customer overview -------------------------
        pdf.add_page()
        pdf.heading("Customer Overview")
        type_mix = sales.groupby("Customer_Type", observed=True)["Invoice_ID"].nunique()
        chart_paths["customer_type"] = render_chart_matplotlib(
            {str(k): float(v) for k, v in type_mix.items()}, "pie",
            chart_dir / "customer_type.png", title="Orders by Customer Type",
        )
        pdf.image(str(chart_paths["customer_type"]), x=15, w=120)
        gender_mix = sales.groupby("Gender", observed=True)["Invoice_ID"].nunique()
        chart_paths["gender"] = render_chart_matplotlib(
            {str(k): float(v) for k, v in gender_mix.items()}, "pie",
            chart_dir / "gender.png", title="Orders by Gender",
        )
        pdf.ln(4)
        member_share = type_mix.get("Member", 0) / max(type_mix.sum(), 1) * 100
        membership_counts = customers_df["Membership"].value_counts()
        pdf.paragraph(
            f"Members placed {member_share:.1f}% of all orders. The customer base "
            f"of {len(customers_df):,} breaks down by membership tier as follows: "
            + ", ".join(f"{tier}: {count:,}" for tier, count in membership_counts.items())
            + f". Orders by gender: "
            + ", ".join(f"{g}: {v:,}" for g, v in gender_mix.items())
            + "."
        )

        # ---------------- Page 6: monthly trends ----------------------------
        pdf.add_page()
        pdf.heading("Monthly Trends")
        monthly = sales.groupby(sales["Date"].dt.to_period("M"))["Total"].sum().sort_index()
        monthly_labels = [str(p) for p in monthly.index]
        chart_paths["monthly"] = render_chart_matplotlib(
            dict(zip(monthly_labels, monthly.values)), "line",
            chart_dir / "monthly.png",
            title="Monthly Revenue (full period)", ylabel="Revenue ($)",
        )
        pdf.image(str(chart_paths["monthly"]), x=15, w=180)
        pdf.ln(4)
        recent = list(zip(monthly_labels[-6:], monthly.values[-6:]))
        pdf.table(
            ["Month", "Revenue ($)"],
            [(m, f"${v:,.2f}") for m, v in recent],
            (95, 95),
        )

        # ---------------- Page 7: recommendations ---------------------------
        pdf.add_page()
        pdf.heading("Recommendations")
        recommendations = [
            f"1. Focus marketing on Branch {best_branch} - highest revenue generator "
            f"(${branch_stats.iloc[0].Revenue:,.2f}).",
            f"2. Expand {best_product} inventory - top selling category "
            f"(${line_revenue.iloc[0]:,.2f} revenue).",
            f"3. Investigate {worst_product} - underperforming category "
            f"(${line_revenue.iloc[-1]:,.2f} revenue).",
            f"4. Target weekend promotions - weekend daily sales run "
            f"{weekend_uplift:.1f}% higher than weekdays.",
            f"5. Address {stockouts} stockout products immediately.",
        ]
        for rec in recommendations:
            pdf.paragraph(rec)

        # ---------------- save ----------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = config.REPORT_PATH / f"executive_report_{timestamp}.pdf"
        pdf.output(str(out_path))

        for path in chart_paths.values():
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Chart cleanup skipped for {}: {}", path, exc)
        try:
            chart_dir.rmdir()
        except OSError as exc:
            logger.debug("Chart temp dir {} not removed: {}", chart_dir, exc)

        logger.info("Executive report saved -> {} ({:.0f} KB).", out_path, out_path.stat().st_size / 1024)
        logger.info("=== Executive report generation complete ===")
        return str(out_path)
    except Exception:
        logger.exception("Executive report generation failed")
        raise


if __name__ == "__main__":
    print(generate_executive_report())
