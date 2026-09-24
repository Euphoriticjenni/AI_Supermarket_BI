"""Schema validation for the AI_Supermarket_BI datasets (pandera-based).

Defines one pandera ``DataFrameSchema`` per dataset — ``SalesSchema``,
``ProductSchema`` and ``CustomerSchema`` — mirroring the business rules of
``database/schema.sql``, plus:

* :func:`validate_dataset` — validate one DataFrame against a named schema
  and return a structured report (never raises into caller code).
* :func:`run_validation` — validate all three generated CSVs and log a
  combined summary.

Design notes
------------
* ``coerce=True`` on date/numeric columns normalises CSV round-trip dtypes
  before the checks run.
* Schemas are not ``strict``: the sales CSV legally carries the generator's
  ``Is_Injected_Anomaly`` ground-truth flag, which is not part of the SQL
  schema.
* Injected anomalies (10x quantity / 5x price) stay inside the declared
  bounds (Quantity <= 200, Unit_Price > 0) on purpose — they must remain
  valid rows so the anomaly-detection module can be scored against them.
"""
from typing import Dict, Optional

import pandera as pa
import pandas as pd

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
SalesSchema = pa.DataFrameSchema(
    {
        "Invoice_ID": pa.Column(str, unique=True, nullable=False),
        "Date": pa.Column(pa.DateTime, nullable=False, coerce=True),
        "Branch": pa.Column(str, pa.Check.isin(list(config.BRANCHES.keys())), nullable=False),
        "City": pa.Column(str, nullable=False),
        "Unit_Price": pa.Column(float, pa.Check.gt(0), coerce=True),
        "Quantity": pa.Column(int, pa.Check.in_range(1, 200), coerce=True),
        "Total": pa.Column(float, pa.Check.gt(0), coerce=True),
        "Rating": pa.Column(float, pa.Check.in_range(1, 10), nullable=True, coerce=True),
        "COGS": pa.Column(float, pa.Check.gt(0), coerce=True),
    },
    strict=False,
)

ProductSchema = pa.DataFrameSchema(
    {
        "Product_ID": pa.Column(str, unique=True, nullable=False),
        "Cost": pa.Column(float, pa.Check.gt(0), coerce=True),
        "Selling_Price": pa.Column(float, pa.Check.gt(0), coerce=True),
        "Stock": pa.Column(int, pa.Check.ge(0), coerce=True),
    },
    strict=False,
)

CustomerSchema = pa.DataFrameSchema(
    {
        "Customer_ID": pa.Column(str, unique=True, nullable=False),
        "Age": pa.Column(int, pa.Check.in_range(18, 70), coerce=True),
    },
    strict=False,
)

#: Registry used by :func:`validate_dataset` for name-based lookups.
SCHEMAS: Dict[str, pa.DataFrameSchema] = {
    "sales": SalesSchema,
    "products": ProductSchema,
    "customers": CustomerSchema,
}


def _count_schema_checks(schema: pa.DataFrameSchema) -> int:
    """Count the checks a schema will perform (for report ``total_checks``).

    Counts one dtype/non-null check per column, plus every custom
    ``Check`` attached to a column and every column/dataframe-level
    uniqueness constraint. Defensive by design: returns at least the number
    of columns even if pandera internals differ across versions.
    """
    total = 0
    try:
        for column in schema.columns.values():
            total += 1 + len(column.checks)
            if getattr(column, "unique", False):
                total += 1
        total += len(getattr(schema, "checks", []) or [])
    except Exception:  # pragma: no cover - pandera internal layout changed
        total = max(total, len(schema.columns))
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate_dataset(df: pd.DataFrame, schema_type: str = "sales") -> Dict:
    """Validate a DataFrame against a named schema and return a report.

    The function never raises: validation failures are captured in lazy
    mode (ALL failures collected at once), summarised and returned so the
    caller can decide how to proceed.

    Parameters
    ----------
    df:
        The DataFrame to validate.
    schema_type:
        One of ``"sales"``, ``"products"`` or ``"customers"``.

    Returns
    -------
    dict
        ``{"passed": bool, "total_checks": int, "failures": int,
        "failure_details": DataFrame or None}`` where ``failures`` counts
        distinct failing rows and ``failure_details`` holds pandera's
        per-case failure table.
    """
    key = schema_type.lower()
    if key not in SCHEMAS:
        raise KeyError(f"Unknown schema_type {schema_type!r}. Expected one of {sorted(SCHEMAS)}")
    schema = SCHEMAS[key]
    total_checks = _count_schema_checks(schema)

    report: Dict = {
        "passed": False,
        "total_checks": total_checks,
        "failures": 0,
        "failure_details": None,
    }
    try:
        schema.validate(df, lazy=True)
        report["passed"] = True
        logger.info(
            "Validation PASSED for '{}' ({} rows x {} checks).",
            schema_type, len(df), total_checks,
        )
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases
        report["failure_details"] = failure_cases
        if not failure_cases.empty and "index" in failure_cases.columns:
            report["failures"] = int(failure_cases["index"].nunique())
        else:
            report["failures"] = int(len(failure_cases))
        logger.error(
            "Validation FAILED for '{}': {} failing rows across {} check types.",
            schema_type, report["failures"], failure_cases["check"].nunique() if not failure_cases.empty else 0,
        )
        if not failure_cases.empty:
            summary = failure_cases.groupby(["column", "check"]).size().reset_index(name="rows")
            for row in summary.head(20).itertuples(index=False):
                logger.error("  column='{}' check='{}' -> {} rows", row.column, row.check, row.rows)
    except Exception as exc:  # unexpected — keep the pipeline alive
        logger.exception("Unexpected error while validating '{}': {}", schema_type, exc)
        report["failure_details"] = pd.DataFrame(
            [{"schema_context": "<unexpected>", "check": str(exc), "failure_case": type(exc).__name__}]
        )
        report["failures"] = 1

    return report


def run_validation(datasets: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Dict]:
    """Validate every generated dataset and return the combined reports.

    When ``datasets`` is omitted the three generated CSVs are read from
    ``data/augmented/``.

    Returns
    -------
    dict
        Mapping of dataset name -> report dict (from :func:`validate_dataset`).
    """
    if datasets is None:
        datasets = {
            "sales": pd.read_csv(config.SALES_DATA_FILE),
            "products": pd.read_csv(config.PRODUCTS_DATA_FILE),
            # keep_default_na=False: the literal membership tier "None" must
            # not be parsed as NaN (it is in pandas' default NA list).
            "customers": pd.read_csv(config.CUSTOMERS_DATA_FILE, keep_default_na=False),
        }

    reports: Dict[str, Dict] = {}
    for name, df in datasets.items():
        reports[name] = validate_dataset(df, name)

    passed = sum(1 for r in reports.values() if r["passed"])
    logger.info(
        "Validation summary: {}/{} datasets passed ({} rows checked in total).",
        passed, len(reports), sum(len(d) for d in datasets.values()),
    )
    return reports


if __name__ == "__main__":
    all_reports = run_validation()
    for name, rep in all_reports.items():
        status = "PASS" if rep["passed"] else "FAIL"
        print(f"[{status}] {name}: {rep['failures']} failing rows of "
              f"{rep['total_checks']} checks")
