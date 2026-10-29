"""NexusBI - Full Pipeline Orchestrator.

Usage::

    python scripts/run_pipeline.py

The orchestrator passes DataFrames between stages where possible, records the
duration and status of each stage, and keeps forecasting/reporting failures
from hiding the outputs of earlier successful stages.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _run_stage(name: str, function: Callable[[], Any], results: dict[str, Any]) -> Any:
    """Execute one stage, record timing, and return ``None`` on failure."""
    started = time.perf_counter()
    logger.info("--- {} started ---", name)
    try:
        output = function()
        elapsed = time.perf_counter() - started
        results[name] = {"status": "ok", "seconds": elapsed}
        logger.info("--- {} complete in {:.2f}s ---", name, elapsed)
        return output
    except Exception as exc:
        elapsed = time.perf_counter() - started
        results[name] = {"status": "failed", "seconds": elapsed, "error": str(exc)}
        logger.exception("--- {} failed after {:.2f}s: {} ---", name, elapsed, exc)
        return None


def _load_frame(path: Path, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Load parquet or CSV, returning an empty frame when unavailable."""
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=parse_dates or [])


def _legacy_sales_frame(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Adapt the regenerated lowercase sales contract for legacy modules."""
    if frame is None or frame.empty or "Invoice_ID" in frame.columns:
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    required = {"transaction_id", "date", "time", "category", "unit_price", "quantity", "total_amount", "unit_cost", "profit"}
    if not required.issubset(frame.columns):
        return frame
    sales = frame.copy()
    sales["Invoice_ID"] = sales["transaction_id"]
    sales["Date"] = pd.to_datetime(sales["date"], errors="coerce")
    sales["Time"] = sales["time"].astype(str).str[:5]
    sales["Branch"] = sales["branch"] if "branch" in sales else "NexusBI"
    sales["City"] = sales["branch"] if "branch" in sales else "NexusBI"
    sales["Customer_Type"] = sales["membership_tier"].map(lambda value: "Member" if value else "Normal") if "membership_tier" in sales else "Normal"
    sales["Gender"] = "Unknown"
    sales["Product_Line"] = sales["category"]
    sales["Unit_Price"] = pd.to_numeric(sales["unit_price"], errors="coerce")
    sales["Quantity"] = pd.to_numeric(sales["quantity"], errors="coerce")
    sales["Tax_5pct"] = (pd.to_numeric(sales["unit_cost"], errors="coerce") * sales["Quantity"] * 0.05).round(2)
    sales["Total"] = pd.to_numeric(sales["total_amount"], errors="coerce")
    sales["Payment"] = sales["payment_method"] if "payment_method" in sales else "Unknown"
    sales["COGS"] = (pd.to_numeric(sales["unit_cost"], errors="coerce") * sales["Quantity"]).round(2)
    sales["Gross_Margin_Pct"] = np.where(sales["Total"] != 0, sales["profit"] / sales["Total"] * 100, 0.0)
    sales["Gross_Income"] = pd.to_numeric(sales["profit"], errors="coerce")
    sales["Rating"] = 7.5
    columns = ["Invoice_ID", "Date", "Time", "Branch", "City", "Customer_Type", "Gender", "Product_Line", "Unit_Price", "Quantity", "Tax_5pct", "Total", "Payment", "COGS", "Gross_Margin_Pct", "Gross_Income", "Rating"]
    return sales[columns]


def _legacy_products_frame(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Adapt the regenerated lowercase product contract for inventory/reporting."""
    if frame is None or frame.empty or "Product_ID" in frame.columns:
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    products = frame.rename(columns={"product_id": "Product_ID", "name": "Product_Name", "category": "Category", "supplier": "Supplier", "unit_cost": "Cost", "unit_price": "Selling_Price"}).copy()
    products["Stock"] = 100
    products["Reorder_Level"] = 20
    return products[["Product_ID", "Product_Name", "Category", "Supplier", "Cost", "Selling_Price", "Stock", "Reorder_Level"]]


def _legacy_customers_frame(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Adapt the regenerated lowercase customer contract for the PDF report."""
    if frame is None or frame.empty or "Customer_ID" in frame.columns:
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    customers = frame.rename(columns={"customer_id": "Customer_ID", "preferred_branch": "City", "membership_tier": "Membership", "join_date": "Join_Date"}).copy()
    customers["Gender"] = "Unknown"
    age_map = {"18-24": 21, "25-34": 30, "35-44": 39, "45-54": 49, "55-64": 59, "65+": 70}
    customers["Age"] = customers["age_group"].map(age_map).fillna(35).astype(int)
    customers["Total_Spent"] = 0.0
    customers["Visit_Count"] = (pd.to_numeric(customers["avg_weekly_visits"], errors="coerce").fillna(0) * 52).round().astype(int)
    return customers[["Customer_ID", "Gender", "Age", "City", "Membership", "Join_Date", "Total_Spent", "Visit_Count"]]


def _forecast_data_path() -> Path:
    """Select the requested canonical forecasting input path."""
    for candidate in (
        config.PROCESSED_DATA_PATH / "sales_features.parquet",
        config.PROCESSED_DATA_PATH / "sales_processed.parquet",
        config.SALES_DATA_FILE,
    ):
        if candidate.exists():
            return candidate
    return config.PROCESSED_DATA_PATH / "sales_features.parquet"


def main() -> int:
    """Run generation, validation, processing, analytics, and reporting."""
    config.ensure_directories()
    timing: dict[str, Any] = {}
    logger.info("=== NexusBI full pipeline started ===")

    generated: Any = _run_stage("Generate data", lambda: _generate_data(), timing)
    sales_df = generated[0] if isinstance(generated, tuple) and len(generated) >= 1 else None
    products_df = generated[1] if isinstance(generated, tuple) and len(generated) >= 2 else None
    customers_df = generated[2] if isinstance(generated, tuple) and len(generated) >= 3 else None

    _run_stage("Validate data", lambda: _validate_data(sales_df, products_df, customers_df), timing)
    processed = _run_stage("Preprocess", _preprocess, timing)
    if not isinstance(processed, pd.DataFrame) or processed.empty:
        processed = _load_frame(config.PROCESSED_DATA_PATH / "sales_processed.parquet", parse_dates=["Date"])

    features = _run_stage("Feature engineering", lambda: _feature_engineering(processed), timing)
    if not isinstance(features, pd.DataFrame) or features.empty:
        features = _load_frame(config.PROCESSED_DATA_PATH / "sales_features.parquet", parse_dates=["Date"])
    if features.empty:
        features = processed

    _run_stage("Analytics", lambda: _analytics(features), timing)
    _run_stage("Inventory", lambda: _inventory(products_df, features), timing)
    forecast = _run_stage("Forecasting", lambda: _forecast(_forecast_data_path()), timing)
    _run_stage("Generate report", lambda: _report(sales_df, products_df, customers_df), timing)

    print("\nPipeline summary")
    print("-" * 72)
    for name, details in timing.items():
        status = details["status"].upper()
        suffix = f" - {details['error']}" if status == "FAILED" else ""
        print(f"{name:24s} {status:8s} {details['seconds']:8.2f}s{suffix}")
    print("-" * 72)
    print("Pipeline complete. Run: streamlit run dashboard/app.py")
    logger.info("=== NexusBI full pipeline finished ===")
    # Failures are visible in the summary, but partial outputs remain usable.
    return 0


def _generate_data() -> Any:
    from src.data_generator import generate_all

    return generate_all()


def _validate_data(sales_df: Optional[pd.DataFrame], products_df: Optional[pd.DataFrame], customers_df: Optional[pd.DataFrame]) -> Any:
    from src.data_validation import run_validation

    datasets = None
    if all(isinstance(frame, pd.DataFrame) for frame in (sales_df, products_df, customers_df)):
        datasets = {"sales": sales_df, "products": products_df, "customers": customers_df}
    return run_validation(datasets=datasets)


def _preprocess() -> pd.DataFrame:
    from src.preprocessing import run_preprocessing_pipeline

    try:
        return run_preprocessing_pipeline(source="csv")
    except Exception as exc:
        logger.warning("Legacy preprocessing could not consume the regenerated schema; applying an in-memory adapter: {}", exc)
        from src.preprocessing import fix_data_types, handle_missing_values, remove_duplicates, remove_outliers, save_processed_data, standardize_categories

        raw = pd.read_csv(config.SALES_DATA_FILE)
        adapted = _legacy_sales_frame(raw)
        adapted = standardize_categories(handle_missing_values(fix_data_types(remove_duplicates(adapted))))
        clean, anomalies = remove_outliers(adapted, keep_anomalies=True)
        if not anomalies.empty:
            anomalies.to_csv(config.OUTLIERS_FILE, index=False)
        clean = clean.reset_index(drop=True)
        save_processed_data(clean, "sales_processed")
        clean.to_csv(config.PROCESSED_SALES_CSV, index=False)
        return clean


def _feature_engineering(sales_df: pd.DataFrame) -> pd.DataFrame:
    from src.feature_engineering import run_feature_engineering

    return run_feature_engineering(sales_df, save=True)


def _analytics(sales_df: pd.DataFrame) -> Any:
    from src.analytics import run_full_analytics

    return run_full_analytics(sales_df)


def _inventory(products_df: Optional[pd.DataFrame], sales_df: pd.DataFrame) -> Any:
    from src.inventory import run_inventory_pipeline

    if products_df is None or products_df.empty:
        products_df = pd.read_csv(config.PRODUCTS_DATA_FILE)
    return run_inventory_pipeline(_legacy_products_frame(products_df), _legacy_sales_frame(sales_df))


def _forecast(data_path: Path) -> Any:
    from src.forecasting import run_forecast_pipeline

    return run_forecast_pipeline(data_path=data_path, periods=90)


def _report(sales_df: Optional[pd.DataFrame], products_df: Optional[pd.DataFrame], customers_df: Optional[pd.DataFrame]) -> Any:
    from src.report_generator import generate_executive_report

    return generate_executive_report(
        sales_df=_legacy_sales_frame(sales_df),
        products_df=_legacy_products_frame(products_df),
        customers_df=_legacy_customers_frame(customers_df),
    )


if __name__ == "__main__":
    raise SystemExit(main())
