"""Cleaning & preprocessing pipeline for the AI_Supermarket_BI project.

Turns the raw generated CSVs into analysis-ready parquet artefacts:

1. Load from CSV or the SQLite database (:func:`load_data`).
2. Remove exact duplicate rows (:func:`remove_duplicates`).
3. Fix data types (:func:`fix_data_types`).
4. Impute missing values — numeric median / categorical mode
   (:func:`handle_missing_values`).
5. Standardise category text (:func:`standardize_categories`).
6. Remove IQR outliers while quarantining them separately for the anomaly
   module (:func:`remove_outliers`).
7. Persist to ``data/processed/`` (:func:`save_processed_data`).

Public entry point: :func:`run_preprocessing_pipeline` (used by
``make preprocess``).
"""
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

#: Columns treated as categorical when fixing dtypes / standardising text.
CATEGORICAL_COLS = ["Branch", "City", "Customer_Type", "Gender", "Product_Line", "Payment"]

#: Default columns for the IQR outlier filter.
DEFAULT_OUTLIER_COLUMNS = ["Unit_Price", "Quantity", "Total"]

#: Maps logical table names to the generated CSV files.
TABLE_FILES = {
    "sales": config.SALES_DATA_FILE,
    "products": config.PRODUCTS_DATA_FILE,
    "customers": config.CUSTOMERS_DATA_FILE,
}


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------
def load_data(source: str = "csv", table_name: str = "sales") -> pd.DataFrame:
    """Load a dataset from a CSV file or the SQLite database.

    Parameters
    ----------
    source:
        ``"csv"`` (default) reads ``data/augmented/{table}.csv``;
        ``"db"`` (or ``"database"``) reads the table via SQLAlchemy.
    table_name:
        Logical table name — ``"sales"``, ``"products"`` or ``"customers"``
        (mapped to their real CSV filenames), or any other filename stem
        that exists under ``data/augmented/``.

    Returns
    -------
    pandas.DataFrame
        The loaded data with ``Date`` parsed as datetime. Shape and memory
        usage are logged.
    """
    try:
        if source.lower() in ("db", "database"):
            from sqlalchemy import create_engine

            db_path = Path(config.DATABASE_PATH)
            engine = create_engine(f"sqlite:///{db_path.as_posix()}")
            try:
                df = pd.read_sql_table(table_name, engine)
            finally:
                engine.dispose()
            origin = f"database {db_path}::{table_name}"
        else:
            csv_path = TABLE_FILES.get(table_name, config.AUGMENTED_DATA_PATH / f"{table_name}.csv")
            # keep_default_na=False for customers: the literal membership
            # tier "None" must not be parsed as NaN.
            keep_na = table_name == "customers"
            df = pd.read_csv(csv_path, keep_default_na=keep_na)
            origin = str(csv_path)

        if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Is_Injected_Anomaly" in df.columns:
            df["Is_Injected_Anomaly"] = df["Is_Injected_Anomaly"].astype(bool)

        memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
        logger.info("Loaded {} -> {:,} rows x {} columns ({:.1f} MB).",
                    origin, len(df), df.shape[1], memory_mb)
        return df
    except Exception:
        logger.exception("Failed to load data (source='{}', table_name='{}')", source, table_name)
        raise


def load_processed_or_csv() -> pd.DataFrame:
    """Load the most recent processed sales artefact.

    Prefers the processed parquet, falls back to the processed CSV, then to
    the raw generated CSV when preprocessing has not run yet.
    """
    for candidate in (
        config.PROCESSED_SALES_PARQUET,
        config.PROCESSED_SALES_CSV,
        config.SALES_DATA_FILE,
    ):
        if candidate.exists():
            if candidate.suffix == ".parquet":
                df = pd.read_parquet(candidate)
            else:
                df = pd.read_csv(candidate, parse_dates=["Date"])
            logger.info("Loaded {:,} rows from {}.", len(df), candidate)
            return df
    raise FileNotFoundError(
        "No sales data found — run data generation first (python -m src.data_generator)."
    )


def save_processed_data(df: pd.DataFrame, filename: str) -> Path:
    """Persist a processed DataFrame to ``data/processed/{filename}.parquet``.

    Falls back to CSV when no parquet engine (pyarrow/fastparquet) is
    available — the pipeline must never crash on a missing optional
    dependency (e.g. inside the slim Docker image).

    Returns
    -------
    pathlib.Path
        The path actually written (parquet or CSV fallback); the file size
        is logged either way.
    """
    try:
        config.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        parquet_path = config.PROCESSED_DATA_PATH / f"{filename}.parquet"
        df.to_parquet(parquet_path, index=False)
        target = parquet_path
    except (ImportError, ValueError) as exc:
        csv_path = config.PROCESSED_DATA_PATH / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        target = csv_path
        logger.warning("Parquet engine unavailable ({}); wrote CSV fallback.", type(exc).__name__)
    size_mb = target.stat().st_size / 1024 ** 2
    logger.info("Saved {:,} rows -> {} ({:.2f} MB).", len(df), target, size_mb)
    return target


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and log how many were removed."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.warning("Removed {} duplicate rows ({:.3f}% of {:,}).",
                       removed, 100 * removed / before, before)
    else:
        logger.info("No duplicate rows found across {:,} rows.", before)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values — numeric with the median, categorical with the
    mode — logging every column that needed treatment."""
    null_counts = df.isna().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        logger.info("No missing values detected across {:,} rows.", len(df))
        return df

    for column, count in null_counts.items():
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        if is_numeric:
            fill_value = df[column].median()
        else:
            mode = df[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        df[column] = df[column].fillna(fill_value)
        logger.warning("Column '{}' had {} nulls ({:.2f}%) -> filled with {} = {!r}",
                       column, count, 100 * count / len(df),
                       "median" if is_numeric else "mode", fill_value)
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce canonical dtypes: ``Date`` as datetime64, money/rating columns
    as float64, count columns as int64, categorical columns as category."""
    try:
        if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for column in ("Unit_Price", "Tax_5pct", "Total", "COGS", "Gross_Margin_Pct",
                       "Gross_Income", "Rating", "Cost", "Selling_Price", "Total_Spent"):
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
        for column in ("Quantity", "Stock", "Reorder_Level", "Visit_Count", "Age"):
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").astype("int64")
        for column in CATEGORICAL_COLS:
            if column in df.columns:
                df[column] = df[column].astype(str).astype("category")
        if "Time" in df.columns:
            df["Time"] = df["Time"].astype(str)
        logger.info("Data types fixed (dates parsed, numerics cast, categories encoded).")
        return df
    except Exception:
        logger.exception("fix_data_types failed")
        raise


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    method: str = "iqr",
    keep_anomalies: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove outliers from numeric columns using the IQR rule.

    Values outside ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`` are flagged. Removed
    rows are returned separately as ``anomaly_df`` (the "keep anomalies"
    requirement: the anomaly-detection module compares its findings against
    them, including the generator's ``Is_Injected_Anomaly`` ground truth).

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numeric columns to filter; defaults to
        ``["Unit_Price", "Quantity", "Total"]``.
    method:
        Only ``"iqr"`` is supported.
    keep_anomalies:
        When True (default) outliers are returned in ``anomaly_df`` so they
        are preserved rather than discarded.

    Returns
    -------
    tuple(pandas.DataFrame, pandas.DataFrame)
        ``(clean_df, anomaly_df)``; ``anomaly_df`` is empty when
        ``keep_anomalies=False``.
    """
    if method != "iqr":
        raise ValueError(f"Unsupported outlier method: {method!r}")
    columns = columns if columns is not None else DEFAULT_OUTLIER_COLUMNS
    if df.empty or not columns:
        return df.copy(), df.iloc[0:0].copy()

    mask = pd.Series(False, index=df.index)
    for column in columns:
        if column not in df.columns:
            logger.warning("Outlier column '{}' not present — skipped.", column)
            continue
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        column_mask = ~df[column].between(lower, upper)
        mask |= column_mask
        logger.info("IQR bounds for '{}': [{:.2f}, {:.2f}] -> {} rows outside",
                    column, lower, upper, int(column_mask.sum()))

    anomaly_df = df.loc[mask].copy()
    clean_df = df.loc[~mask].copy()
    if "Is_Injected_Anomaly" in df.columns:
        logger.info("Injected anomalies quarantined: {}/{}.",
                    int(anomaly_df["Is_Injected_Anomaly"].sum()),
                    int(df["Is_Injected_Anomaly"].sum()))
    logger.warning("Outlier removal: {} of {:,} rows flagged ({:.2f}%).",
                   len(anomaly_df), len(df), 100 * len(anomaly_df) / max(len(df), 1))

    if not keep_anomalies:
        anomaly_df = df.iloc[0:0].copy()
    return clean_df, anomaly_df


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise category text.

    Strips/collapses whitespace on all string columns, then title-cases
    ``Product_Line``, ``City``, ``Payment``, ``Gender`` and
    ``Customer_Type``. Title-cased values are re-mapped to the canonical
    spellings defined in ``src/config.py`` (e.g. "Health And Beauty" ->
    "Health and beauty", "Credit Card" -> "Credit card") so downstream
    filters and SQL aggregates keep matching.
    """
    canonical = {
        "Health And Beauty": "Health and beauty",
        "Electronic Accessories": "Electronic accessories",
        "Home And Lifestyle": "Home and lifestyle",
        "Sports And Travel": "Sports and travel",
        "Food And Beverages": "Food and beverages",
        "Fashion Accessories": "Fashion accessories",
        "Credit Card": "Credit card",
        "Ewallet": "Ewallet",
        "Cash": "Cash",
    }
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype(str).str.strip().str.split().str.join(" ")

    for column in ("Product_Line", "City", "Payment", "Gender", "Customer_Type"):
        if column in df.columns:
            df[column] = df[column].str.title().replace(canonical)
    logger.info("Category columns standardised (stripped, title-cased, canonicalised).")
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_preprocessing_pipeline(source: str = "csv") -> pd.DataFrame:
    """Run the complete cleaning pipeline and persist the artefacts.

    Chain: load -> deduplicate -> fix types -> impute missing values ->
    standardise categories -> IQR outlier removal (outliers quarantined to
    ``outliers_removed.csv``) -> save processed data.

    Returns
    -------
    pandas.DataFrame
        The clean, analysis-ready sales DataFrame.
    """
    logger.info("=== Preprocessing pipeline started (source='{}') ===", source)
    try:
        df = load_data(source=source, table_name="sales")
        df = remove_duplicates(df)
        df = fix_data_types(df)
        df = handle_missing_values(df)
        df = standardize_categories(df)

        clean_df, anomaly_df = remove_outliers(df, keep_anomalies=True)
        if not anomaly_df.empty:
            anomaly_df.to_csv(config.OUTLIERS_FILE, index=False)
            logger.info("Quarantined outliers saved -> {}", config.OUTLIERS_FILE)

        clean_df = clean_df.reset_index(drop=True)
        save_processed_data(clean_df, "sales_processed")
        # A CSV twin helps environments without a parquet engine (Docker).
        clean_df.to_csv(config.PROCESSED_SALES_CSV, index=False)

        logger.info(
            "=== Preprocessing complete: {:,} clean rows x {} columns, {} outliers quarantined ===",
            len(clean_df), clean_df.shape[1], len(anomaly_df),
        )
        return clean_df
    except Exception:
        logger.exception("Preprocessing pipeline failed")
        raise


if __name__ == "__main__":
    run_preprocessing_pipeline()
