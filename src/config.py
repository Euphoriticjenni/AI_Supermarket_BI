"""Centralized configuration for the AI_Supermarket_BI project.

Every module imports its constants from here so that paths, seeds, business
parameters and styling stay consistent across data generation, modelling,
the Streamlit dashboard and the report generator.

Environment variables (loaded from a ``.env`` file when present) override the
defaults below, which keeps the project fully configurable without touching
code::

    cp .env.example .env

Notes
-----
* All paths are anchored to ``PROJECT_ROOT`` (the folder that contains
  ``src/``), so the project behaves identically whether it is driven from
  the Makefile, pytest, Streamlit or Docker.
* ``NUM_RECORDS`` can be lowered through the environment (e.g.
  ``NUM_RECORDS=20000``) for quick smoke tests without editing code.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    # python-dotenv is an optional convenience: with it missing, plain
    # environment variables still work and only the .env file auto-load
    # below is skipped, so the app keeps booting in minimal environments.
    load_dotenv = None

# ---------------------------------------------------------------------------
# Project root & environment
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Load .env (if present). Real environment variables keep precedence over
#: values defined in the file, and .env values override the code defaults.
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Core generation constants
# ---------------------------------------------------------------------------
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

#: Number of transaction lines to synthesise (env-overridable for smoke tests).
NUM_RECORDS = int(os.getenv("NUM_RECORDS", "500000"))

DATE_START = "2022-01-01"
DATE_END = "2023-12-31"

BRANCHES = {"A": "Yangon", "B": "Mandalay", "C": "Naypyitaw"}

PRODUCT_LINES = [
    "Health and beauty",
    "Electronic accessories",
    "Home and lifestyle",
    "Sports and travel",
    "Food and beverages",
    "Fashion accessories",
]

PAYMENT_METHODS = ["Cash", "Credit card", "Ewallet"]

CUSTOMER_TYPES = ["Member", "Normal"]

GENDERS = ["Male", "Female"]

MEMBERSHIP_TIERS = ["Gold", "Silver", "Bronze", "None"]

# ---------------------------------------------------------------------------
# Derived business constants
# ---------------------------------------------------------------------------
CITIES = sorted(set(BRANCHES.values()))

#: Sales tax applied at the checkout (5%% of COGS).
TAX_RATE = 0.05

#: Fixed gross margin percentage of the retail chain (5 / 105 * 100).
GROSS_MARGIN_PCT = 4.7619

#: Number of extreme transactions injected by the data generator so the
#: anomaly-detection module has a ground truth to validate against.
ANOMALIES_TO_INJECT = 50

#: Service-level Z-score (1.65 => 95%%) used for safety-stock calculations.
SERVICE_LEVEL_Z = 1.65

#: Supplier lead time in days used by the inventory reorder logic.
LEAD_TIME_DAYS = 7


def _resolve_path(env_key: str, default: str) -> Path:
    """Resolve a configurable path.

    Reads ``env_key`` from the environment (usually set via ``.env``); when
    the value is relative it is anchored to the project root so the project
    works from any working directory.
    """
    raw = os.getenv(env_key, default)
    path = Path(raw)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
DATABASE_PATH = _resolve_path("DATABASE_PATH", "database/supermarket.db")
RAW_DATA_PATH = _resolve_path("RAW_DATA_PATH", "data/raw")
PROCESSED_DATA_PATH = _resolve_path("PROCESSED_DATA_PATH", "data/processed")
AUGMENTED_DATA_PATH = _resolve_path("AUGMENTED_DATA_PATH", "data/augmented")
MODEL_REGISTRY_PATH = _resolve_path("MODEL_REGISTRY_PATH", "models/registry")
REPORT_PATH = _resolve_path("REPORT_PATH", "reports")
LOG_PATH = _resolve_path("LOG_PATH", "logs")

# Raw / augmented (generated) datasets -------------------------------------
SALES_DATA_FILE = AUGMENTED_DATA_PATH / "supermarket_sales.csv"
PRODUCTS_DATA_FILE = AUGMENTED_DATA_PATH / "products.csv"
CUSTOMERS_DATA_FILE = AUGMENTED_DATA_PATH / "customers.csv"
INJECTED_ANOMALIES_FILE = AUGMENTED_DATA_PATH / "injected_anomalies.csv"

# Processed artefacts -------------------------------------------------------
PROCESSED_SALES_PARQUET = PROCESSED_DATA_PATH / "sales_processed.parquet"
PROCESSED_SALES_CSV = PROCESSED_DATA_PATH / "sales_processed.csv"
PROCESSED_PRODUCTS_FILE = PROCESSED_DATA_PATH / "products_processed.csv"
PROCESSED_CUSTOMERS_FILE = PROCESSED_DATA_PATH / "customers_processed.csv"
OUTLIERS_FILE = PROCESSED_DATA_PATH / "outliers_removed.csv"
SEGMENTS_FILE = PROCESSED_DATA_PATH / "customer_segments.csv"
ASSOCIATION_RULES_FILE = PROCESSED_DATA_PATH / "association_rules.csv"
ANOMALIES_FILE = PROCESSED_DATA_PATH / "detected_anomalies.csv"
REORDER_REPORT_FILE = PROCESSED_DATA_PATH / "reorder_report.csv"
ABC_ANALYSIS_FILE = PROCESSED_DATA_PATH / "abc_analysis.csv"
MODEL_COMPARISON_FILE = PROCESSED_DATA_PATH / "model_comparison.csv"
FORECAST_FILE = PROCESSED_DATA_PATH / "forecast_next_30d.csv"

# ---------------------------------------------------------------------------
# Logging / assistant settings
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

#: Optional OpenAI key. When empty the AI assistant runs in offline,
#: rule-based mode (the default and fully supported path).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# ---------------------------------------------------------------------------
# Holidays (used by feature engineering & Prophet)
# ---------------------------------------------------------------------------
_FIXED_HOLIDAYS = {
    "New Year's Day": (1, 1),
    "Valentine's Day": (2, 14),
    "Labour Day": (5, 1),
    "Halloween": (10, 31),
    "Christmas Eve": (12, 24),
    "Christmas Day": (12, 25),
    "New Year's Eve": (12, 31),
}
_EASTER_SUNDAY = {2022: (4, 17), 2023: (4, 9), 2024: (3, 31)}

HOLIDAYS = {}
for _year in (2022, 2023, 2024):
    for _name, (_month, _day) in _FIXED_HOLIDAYS.items():
        HOLIDAYS[f"{_year:04d}-{_month:02d}-{_day:02d}"] = _name
    for _day in range(13, 17):
        HOLIDAYS[f"{_year:04d}-04-{_day:02d}"] = "Thingyan Water Festival"
    _easter_month, _easter_day = _EASTER_SUNDAY[_year]
    HOLIDAYS[f"{_year:04d}-{_easter_month:02d}-{_easter_day:02d}"] = "Easter Sunday"

# ---------------------------------------------------------------------------
# Shared chart styling (consistent palette across every dashboard page)
# ---------------------------------------------------------------------------
#: Qualitative "Set2" palette used by all plotly charts in the project.
COLOR_PALETTE = [
    "#66C2A5",  # teal green
    "#FC8D62",  # orange
    "#8DA0CB",  # slate blue
    "#E78AC3",  # pink
    "#A6D854",  # lime
    "#FFD92F",  # yellow
]

PLOTLY_TEMPLATE = "plotly_white"

CHART_HEIGHT = 420


def ensure_directories() -> None:
    """Create every runtime directory the pipeline writes to.

    Idempotent: safe to call at any point (missing parents are created and
    existing folders are left untouched).
    """
    for directory in (
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        AUGMENTED_DATA_PATH,
        MODEL_REGISTRY_PATH,
        REPORT_PATH,
        LOG_PATH,
        DATABASE_PATH.parent,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Tiny self-check: print the resolved configuration and create folders.
    ensure_directories()
    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Random seed       : {RANDOM_SEED}")
    print(f"Records to create : {NUM_RECORDS:,}")
    print(f"Date range        : {DATE_START} -> {DATE_END}")
    print(f"Branches          : {BRANCHES}")
    print(f"Database          : {DATABASE_PATH}")
    print(f"Raw data dir      : {RAW_DATA_PATH}")
    print(f"Processed dir     : {PROCESSED_DATA_PATH}")
    print(f"Augmented dir     : {AUGMENTED_DATA_PATH}")
    print(f"Model registry    : {MODEL_REGISTRY_PATH}")
    print(f"Reports dir       : {REPORT_PATH}")
    print(f"Logs dir          : {LOG_PATH}")
    print(f"Holidays loaded   : {len(HOLIDAYS)}")
    print("All runtime directories are ready.")
