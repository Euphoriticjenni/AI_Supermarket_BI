"""Print a diagnostic report for the NexusBI augmented datasets.

Run from the project root with::

    python scripts/audit_data.py

The report is intentionally text-only so it is useful in CI logs as well as
from a developer terminal.  It accepts both the current lowercase schema and
the older uppercase CSVs that may still be present before regeneration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "augmented"


def _column(frame: pd.DataFrame, *names: str) -> Optional[str]:
    """Find a column case-insensitively, including legacy aliases."""
    lookup = {str(name).lower(): name for name in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return str(lookup[name.lower()])
    return None


def _money_column(sales: pd.DataFrame) -> Optional[str]:
    return _column(sales, "total_amount", "Total", "revenue", "Revenue")


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _print_frame_profile(label: str, frame: pd.DataFrame) -> None:
    """Print row count, columns, dtype, and null count for one dataset."""
    print(f"\n{label}: {len(frame):,} rows")
    if frame.empty:
        print("  File missing or empty.")
        return
    for name in frame.columns:
        print(f"  {name:28s} dtype={str(frame[name].dtype):12s} nulls={int(frame[name].isna().sum()):,}")


def _print_distribution(title: str, values: pd.Series) -> None:
    print(f"\n{title}")
    counts = values.astype("string").fillna("<null>").value_counts(dropna=False)
    total = max(int(counts.sum()), 1)
    for value, count in counts.items():
        print(f"  {str(value):30s} {int(count):>10,} ({count / total * 100:6.2f}%)")


def _print_monthly_bars(sales: pd.DataFrame, revenue_column: str, date_column: str) -> None:
    print("\nMonthly revenue trend")
    dates = pd.to_datetime(sales[date_column], errors="coerce")
    monthly = sales.assign(_month=dates.dt.to_period("M")).groupby("_month", observed=True)[revenue_column].sum()
    if monthly.empty:
        print("  No valid dates available.")
        return
    maximum = float(monthly.abs().max()) or 1.0
    for month, value in monthly.items():
        width = int(abs(float(value)) / maximum * 42)
        sign = "-" if value < 0 else " "
        print(f"  {str(month)} {sign}{'█' * width:<42s} ${value:,.2f}")


def _variety_score(sales: pd.DataFrame) -> int:
    category = _column(sales, "category", "Product_Line")
    branch = _column(sales, "branch", "Branch")
    payment = _column(sales, "payment_method", "Payment")
    date = _column(sales, "date", "Date", "datetime")
    score = 100
    penalties: list[str] = []
    category_count = int(sales[category].nunique()) if category else 0
    branch_count = int(sales[branch].nunique()) if branch else 0
    payment_count = int(sales[payment].nunique()) if payment else 0
    if category_count < 5:
        score -= 20
        penalties.append("fewer than 5 categories")
    if branch_count < 3:
        score -= 20
        penalties.append("fewer than 3 branches")
    if payment_count < 3:
        score -= 20
        penalties.append("fewer than 3 payment methods")

    seasonal = False
    if date:
        dates = pd.to_datetime(sales[date], errors="coerce")
        revenue_column = _money_column(sales)
        if revenue_column:
            monthly = sales.assign(_month=dates.dt.month).groupby("_month", observed=True)[revenue_column].sum()
            if len(monthly) >= 6 and float(monthly.mean()) != 0:
                seasonal = float(monthly.std() / abs(monthly.mean())) >= 0.08
    if not seasonal:
        score -= 30
        penalties.append("no measurable monthly seasonality")
    print(f"\nData variety score: {max(0, score)}/100")
    print("  Seasonality detected:", "yes" if seasonal else "no")
    print("  Penalties:", "; ".join(penalties) if penalties else "none")
    return max(0, score)


def audit_data() -> int:
    """Read all available augmented CSVs and print the complete audit."""
    sales_path = DATA_DIR / "supermarket_sales.csv"
    products_path = DATA_DIR / "products.csv"
    customers_path = DATA_DIR / "customers.csv"
    if not sales_path.exists():
        logger.error("Sales file not found: {}", sales_path)
        return 1

    sales = pd.read_csv(sales_path)
    products = pd.read_csv(products_path) if products_path.exists() else pd.DataFrame()
    customers = pd.read_csv(customers_path, keep_default_na=False) if customers_path.exists() else pd.DataFrame()
    _print_header("NexusBI DATA AUDIT")
    print(f"Project root: {ROOT}")
    print(f"Sales file:   {sales_path} ({sales_path.stat().st_size / 1024 ** 2:.1f} MB)")
    print(f"Rows:         {len(sales):,}")

    _print_header("Row counts, columns, dtypes, and null counts")
    _print_frame_profile("Sales", sales)
    _print_frame_profile("Products", products)
    _print_frame_profile("Customers", customers)

    date_column = _column(sales, "datetime", "date", "Date")
    parsed_dates = pd.to_datetime(sales[date_column], errors="coerce") if date_column else pd.Series(dtype="datetime64[ns]")
    _print_header("Date range")
    if parsed_dates.notna().any():
        print(f"  Minimum: {parsed_dates.min()}")
        print(f"  Maximum: {parsed_dates.max()}")
    else:
        print("  No parseable dates found.")

    _print_header("Categorical variety")
    for name in sales.select_dtypes(include=["object", "category", "bool"]).columns:
        unique = sales[name].drop_duplicates().astype(str).tolist()
        preview = ", ".join(unique[:20])
        suffix = " ..." if len(unique) > 20 else ""
        print(f"  {name:28s} unique={len(unique):,}: {preview}{suffix}")

    revenue_column = _money_column(sales)
    _print_header("Revenue distribution")
    if revenue_column:
        revenue = pd.to_numeric(sales[revenue_column], errors="coerce").dropna()
        print(f"  Column: {revenue_column}")
        print(f"  Mean:   ${revenue.mean():,.2f}")
        print(f"  Median: ${revenue.median():,.2f}")
        print(f"  Std:    ${revenue.std():,.2f}")
        print(f"  Min:    ${revenue.min():,.2f}")
        print(f"  Max:    ${revenue.max():,.2f}")
        print(f"  Skew:   {revenue.skew():,.3f}")
    else:
        print("  Revenue column not found.")

    product_id = _column(sales, "product_id", "Product_ID")
    product_name = _column(sales, "product_name", "Product_Name")
    customer_id = _column(sales, "customer_id", "Customer_ID")
    if revenue_column and product_id:
        _print_header("Top 10 products by revenue")
        group_columns = [product_id] + ([product_name] if product_name else [])
        top_products = sales.groupby(group_columns, dropna=False)[revenue_column].sum().sort_values(ascending=False).head(10)
        for key, value in top_products.items():
            label = " / ".join(map(str, key if isinstance(key, tuple) else (key,)))
            print(f"  {label:42s} ${value:,.2f}")
    if revenue_column and customer_id:
        _print_header("Top 10 customers by spend")
        top_customers = sales.groupby(customer_id, dropna=False)[revenue_column].sum().sort_values(ascending=False).head(10)
        for key, value in top_customers.items():
            print(f"  {str(key):42s} ${value:,.2f}")

    if revenue_column and date_column:
        _print_monthly_bars(sales, revenue_column, date_column)
    day_column = _column(sales, "day_of_week", "Day_Name")
    if day_column:
        _print_distribution("Day-of-week distribution", sales[day_column])
    elif parsed_dates.notna().any():
        _print_distribution("Day-of-week distribution", parsed_dates.dt.day_name())
    hour_column = _column(sales, "hour_of_day", "Purchase_Hour")
    if hour_column:
        _print_distribution("Hour-of-day distribution", sales[hour_column])
    elif date_column:
        _print_distribution("Hour-of-day distribution", parsed_dates.dt.hour)

    category_column = _column(sales, "category", "Product_Line")
    if category_column and revenue_column:
        _print_header("Category revenue breakdown")
        category_totals = sales.groupby(category_column, dropna=False)[revenue_column].sum().sort_values(ascending=False)
        total = category_totals.sum() or 1.0
        for key, value in category_totals.items():
            print(f"  {str(key):30s} ${value:,.2f} ({value / total * 100:6.2f}%)")
    payment_column = _column(sales, "payment_method", "Payment")
    if payment_column:
        _print_distribution("Payment method breakdown", sales[payment_column])

    _print_header("Anomaly checks")
    price_column = _column(sales, "unit_price", "Unit_Price")
    quantity_column = _column(sales, "quantity", "Quantity")
    transaction_column = _column(sales, "transaction_id", "Invoice_ID")
    future_count = int((parsed_dates > pd.Timestamp.now()).sum()) if parsed_dates.notna().any() else 0
    print(f"  Negative prices:       {int((pd.to_numeric(sales[price_column], errors='coerce') < 0).sum()) if price_column else 'n/a':>10}")
    print(f"  Zero quantities:       {int((pd.to_numeric(sales[quantity_column], errors='coerce') == 0).sum()) if quantity_column else 'n/a':>10}")
    print(f"  Future dates:          {future_count:>10,}")
    print(f"  Duplicate transaction IDs: {int(sales[transaction_column].duplicated().sum()) if transaction_column else 'n/a':>6}")
    print(f"  Duplicate complete rows:   {int(sales.duplicated().sum()):>6,}")
    _variety_score(sales)
    print("\nAudit complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(audit_data())
