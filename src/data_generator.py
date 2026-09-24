"""Synthetic data generation for the AI_Supermarket_BI project.

Builds three realistic, inter-related datasets with Faker + numpy + pandas:

1. ``supermarket_sales.csv``  (~500,000 rows, 2022-01-01 -> 2023-12-31)
   with seasonal patterns, hourly peaks, weighted product lines and 50
   injected anomalies for the anomaly-detection module.
2. ``products.csv``           (~200 products across the 6 product lines)
3. ``customers.csv``          (~10,000 customers with membership tiers)

All artefacts are written to ``data/augmented/`` and loaded into the SQLite
database defined by ``database/schema.sql``.

Usage (from the project root)::

    python -m src.data_generator            # or: make generate-data
"""
import sqlite3
import uuid
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Static generation pools
# ---------------------------------------------------------------------------

#: Realistic product-name pools per category (>= 15 names each as required).
PRODUCT_NAME_POOLS: Dict[str, list] = {
    "Health and beauty": [
        "Face Cream", "Herbal Shampoo", "Lipstick", "Vitamin C Serum",
        "Sunscreen SPF 50", "Toothpaste Pack", "Hand Sanitizer", "Body Lotion",
        "Pain Relief Balm", "Electric Toothbrush", "Hair Conditioner",
        "Face Mask Sheet", "Deodorant Stick", "Razor Blade Pack",
        "Multivitamin Bottle", "Essential Oil Set", "Lip Balm",
        "Makeup Remover", "Aftershave Lotion", "Nail Care Kit",
    ],
    "Electronic accessories": [
        "USB Cable", "Headphones", "Wireless Mouse", "Bluetooth Speaker",
        "Power Bank", "Phone Case", "Screen Protector", "Memory Card",
        "Wall Charger", "LED Desk Lamp", "HDMI Cable", "Wireless Keyboard",
        "Earphone Splitter", "Laptop Stand", "USB Hub", "Smartwatch Band",
        "Car Phone Mount", "Mini Tripod", "Extension Cord", "Batteries Pack",
    ],
    "Home and lifestyle": [
        "Pillow", "Table Lamp", "Curtain Set", "Scented Candle",
        "Bed Sheet Set", "Towel Set", "Kitchen Knife Set", "Cutting Board",
        "Storage Box", "Photo Frame", "Throw Blanket", "Door Mat",
        "Laundry Basket", "Dinnerware Set", "Cooking Pot", "Water Pitcher",
        "Bath Mat", "Wall Clock", "Plant Pot", "Air Freshener",
    ],
    "Sports and travel": [
        "Yoga Mat", "Water Bottle", "Backpack", "Dumbbell Set",
        "Resistance Band", "Jump Rope", "Running Socks", "Gym Gloves",
        "Swim Goggles", "Badminton Racket", "Camping Lantern", "Travel Pillow",
        "Hiking Cap", "Sport Duffel Bag", "Tennis Ball Pack", "Cycling Helmet",
        "Massage Ball", "Shoe Bag", "Insulated Lunch Box", "Fitness Tracker Band",
    ],
    "Food and beverages": [
        "Organic Coffee", "Green Tea", "Protein Bar", "Instant Noodle Pack",
        "Chocolate Box", "Olive Oil Bottle", "Honey Jar", "Fruit Jam",
        "Basmati Rice Bag", "Pasta Pack", "Orange Juice", "Mineral Water Pack",
        "Potato Chips", "Breakfast Cereal", "Dried Mango Pack", "Peanut Butter",
        "Coconut Milk Can", "Spice Mix Box", "Biscuit Tin", "Energy Drink Pack",
    ],
    "Fashion accessories": [
        "Leather Belt", "Sunglasses", "Watch", "Silk Scarf",
        "Baseball Cap", "Canvas Tote Bag", "Handbag", "Ankle Socks Pack",
        "Hair Clip Set", "Beanie Hat", "Leather Wallet", "Necklace Set",
        "Earring Set", "Bracelet", "Neck Tie", "Gloves Pair",
        "Backpack Purse", "Bucket Hat", "Ring Set", "Shawl Wrap",
    ],
}

#: Prefixes used to keep product names unique once a pool is exhausted.
_NAME_MODIFIERS = [
    "Classic", "Premium", "Value", "Deluxe",
    "Eco", "Compact", "Signature", "Essential",
]

#: Basket quantity probabilities for quantities 1..10 (sums to 1.0).
QUANTITY_WEIGHTS = [0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03]

#: Product-line weights: Food and beverages is the best seller (~25%).
PRODUCT_LINE_WEIGHTS = {
    "Health and beauty": 0.15,
    "Electronic accessories": 0.15,
    "Home and lifestyle": 0.15,
    "Sports and travel": 0.15,
    "Food and beverages": 0.25,
    "Fashion accessories": 0.15,
}

#: Store opening hours used when clipping the hour-of-day mixture.
HOUR_MIN, HOUR_MAX = 8, 22


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seed_everything(seed: int) -> Faker:
    """Seed numpy and Faker so every run is fully reproducible.

    Parameters
    ----------
    seed:
        The seed value (``config.RANDOM_SEED`` by default).

    Returns
    -------
    Faker
        A seeded Faker instance ready for deterministic generation.
    """
    np.random.seed(seed)
    Faker.seed(seed)
    return Faker()


def _unique_invoice_ids(n: int) -> np.ndarray:
    """Generate ``n`` guaranteed-unique invoice IDs ("INV-XXXXXX").

    ``uuid4().hex[:6].upper()`` draws from ~16.7M possibilities, so with
    500K rows the birthday paradox produces a few thousand collisions; this
    helper simply re-draws the collided portion until every ID is unique
    (required by the ``sales`` PRIMARY KEY).
    """
    collected: list = []
    seen: set = set()
    while len(collected) < n:
        needed = n - len(collected)
        # +5% slack absorbs the expected collision rate in one extra pass.
        batch = [f"INV-{uuid.uuid4().hex[:6].upper()}" for _ in range(int(needed * 1.05) + 16)]
        for invoice_id in batch:
            if invoice_id not in seen:
                seen.add(invoice_id)
                collected.append(invoice_id)
                if len(collected) == n:
                    break
    return np.array(collected, dtype=object)


def _build_date_probabilities() -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Return the date index and normalised sampling probabilities.

    Implements the seasonal business rules:

    * December:  +20%  (holiday shopping)
    * March:     +15%  (spring sales)
    * February:  -10%  (post-holiday dip)
    * Weekends:  +25%  versus weekdays
    """
    dates = pd.date_range(start=config.DATE_START, end=config.DATE_END, freq="D")
    weights = np.ones(len(dates), dtype=float)
    weights[dates.month == 12] *= 1.20
    weights[dates.month == 3] *= 1.15
    weights[dates.month == 2] *= 0.90
    weights[dates.dayofweek >= 5] *= 1.25
    probabilities = weights / weights.sum()
    return dates, probabilities


def _format_times(hours_float: np.ndarray) -> np.ndarray:
    """Convert fractional hours into "HH:MM" strings with random minutes."""
    hours = np.clip(hours_float, HOUR_MIN, HOUR_MAX).astype(int)
    minutes = np.random.randint(0, 60, size=len(hours))
    return np.array([f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)], dtype=object)


def _sample_hours(n: int) -> np.ndarray:
    """Draw ``n`` hours from a mixture of 3 normals peaking at 10/13/17."""
    component = np.random.choice([0, 1, 2], size=n, p=[0.30, 0.25, 0.45])
    means = np.array([10.0, 13.0, 17.0])[component]
    stds = np.array([1.5, 1.0, 1.5])[component]
    return np.random.normal(loc=means, scale=stds)


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------
def generate_sales_data(n: int = None) -> pd.DataFrame:
    """Generate the synthetic supermarket sales dataset.

    The dataset follows the classic supermarket sales schema (invoice-level
    rows) enriched with realistic seasonality, hour-of-day peaks, weighted
    product lines and exactly ``config.ANOMALIES_TO_INJECT`` extreme
    transactions flagged in a boolean ``Is_Injected_Anomaly`` column (kept in
    the CSV; excluded from the database load to match schema.sql).

    Parameters
    ----------
    n:
        Number of rows to generate. Defaults to ``config.NUM_RECORDS``.

    Returns
    -------
    pandas.DataFrame
        The generated sales data with 18 columns.
    """
    n = int(n) if n else config.NUM_RECORDS
    _seed_everything(config.RANDOM_SEED)
    logger.info("Generating {:,} sales rows spanning {} -> {} ...", n, config.DATE_START, config.DATE_END)

    dates, date_probabilities = _build_date_probabilities()
    date_strings = np.datetime_as_string(dates.values.astype("datetime64[D]"))
    branch_codes = np.array(list(config.BRANCHES.keys()))
    branch_cities = np.array([config.BRANCHES[b] for b in branch_codes])
    lines = np.array(config.PRODUCT_LINES)
    line_probabilities = np.array([PRODUCT_LINE_WEIGHTS[line] for line in lines])
    line_probabilities = line_probabilities / line_probabilities.sum()
    invoice_ids = _unique_invoice_ids(n)

    chunks = []
    chunk_size = 100_000
    generated = 0
    while generated < n:
        size = min(chunk_size, n - generated)

        date_idx = np.random.choice(len(dates), size=size, p=date_probabilities)
        branch_idx = np.random.randint(0, len(branch_codes), size=size)
        line_idx = np.random.choice(len(lines), size=size, p=line_probabilities)

        chunk = pd.DataFrame({
            "Invoice_ID": invoice_ids[generated:generated + size],
            "Date": date_strings[date_idx],
            "Time": _format_times(_sample_hours(size)),
            "Branch": branch_codes[branch_idx],
            "City": branch_cities[branch_idx],
            "Customer_Type": np.random.choice(config.CUSTOMER_TYPES, size=size, p=[0.65, 0.35]),
            "Gender": np.random.choice(config.GENDERS, size=size, p=[0.5, 0.5]),
            "Product_Line": lines[line_idx],
            "Unit_Price": np.round(np.random.uniform(10.0, 100.0, size=size), 2),
            "Quantity": np.random.choice(np.arange(1, 11), size=size, p=QUANTITY_WEIGHTS),
            "Payment": np.random.choice(config.PAYMENT_METHODS, size=size),
        })

        chunk["COGS"] = np.round(chunk["Unit_Price"] * chunk["Quantity"], 2)
        chunk["Tax_5pct"] = np.round(chunk["COGS"] * config.TAX_RATE, 2)
        chunk["Total"] = np.round(chunk["COGS"] + chunk["Tax_5pct"], 2)
        chunk["Gross_Margin_Pct"] = 4.76
        chunk["Gross_Income"] = chunk["Tax_5pct"]
        chunk["Rating"] = np.round(
            np.clip(np.random.normal(7.0, 1.5, size=size), 1.0, 10.0), 1
        )

        chunks.append(chunk)
        generated += size
        logger.info("Generated {:,}/{} sales rows ...", generated, n)

    sales = pd.concat(chunks, ignore_index=True)

    # ---- anomaly injection (ground truth for the anomaly module) ---------
    sales["Is_Injected_Anomaly"] = False
    anomaly_count = min(config.ANOMALIES_TO_INJECT, n)
    anomaly_idx = np.random.choice(n, size=anomaly_count, replace=False)
    sales.loc[anomaly_idx, "Quantity"] = sales.loc[anomaly_idx, "Quantity"] * 10
    sales.loc[anomaly_idx, "Unit_Price"] = np.round(sales.loc[anomaly_idx, "Unit_Price"] * 5, 2)
    sales.loc[anomaly_idx, "COGS"] = np.round(
        sales.loc[anomaly_idx, "Unit_Price"] * sales.loc[anomaly_idx, "Quantity"], 2
    )
    sales.loc[anomaly_idx, "Tax_5pct"] = np.round(sales.loc[anomaly_idx, "COGS"] * config.TAX_RATE, 2)
    sales.loc[anomaly_idx, "Total"] = np.round(
        sales.loc[anomaly_idx, "COGS"] + sales.loc[anomaly_idx, "Tax_5pct"], 2
    )
    sales.loc[anomaly_idx, "Gross_Income"] = sales.loc[anomaly_idx, "Tax_5pct"]
    sales.loc[anomaly_idx, "Is_Injected_Anomaly"] = True
    logger.info("Injected {} anomalous transactions (10x quantity, 5x price).", anomaly_count)

    sales = sales[[
        "Invoice_ID", "Date", "Time", "Branch", "City", "Customer_Type",
        "Gender", "Product_Line", "Unit_Price", "Quantity", "Tax_5pct",
        "Total", "Payment", "COGS", "Gross_Margin_Pct", "Gross_Income",
        "Rating", "Is_Injected_Anomaly",
    ]]
    logger.info("Sales dataset ready: {:,} rows x {} columns.", len(sales), sales.shape[1])
    return sales


def generate_products_data(n: int = 200) -> pd.DataFrame:
    """Generate the product catalogue.

    Each product belongs to one of the six product lines, has a supplier
    code, cost, selling price (cost x 1.2-1.8), stock level and reorder
    level. Ten products are intentionally left at zero stock so the
    inventory module has genuine stockouts to flag.

    Parameters
    ----------
    n:
        Number of products to create (default 200).

    Returns
    -------
    pandas.DataFrame
        The generated product catalogue.
    """
    n = int(n)
    fake = _seed_everything(config.RANDOM_SEED + 1)
    logger.info("Generating {} products ...", n)

    # Distribute products as evenly as possible across the 6 categories.
    per_category = {line: n // len(config.PRODUCT_LINES) for line in config.PRODUCT_LINES}
    for line in config.PRODUCT_LINES[: n % len(config.PRODUCT_LINES)]:
        per_category[line] += 1

    used_names: set = set()
    rows = []
    for category, count in per_category.items():
        pool = PRODUCT_NAME_POOLS[category]
        for i in range(count):
            base_name = pool[i % len(pool)]
            product_name = base_name
            modifier_round = 0
            while product_name in used_names:
                modifier = _NAME_MODIFIERS[modifier_round % len(_NAME_MODIFIERS)]
                suffix = modifier_round // len(_NAME_MODIFIERS) + 1
                product_name = f"{modifier} {base_name}" + (f" {suffix}" if suffix > 1 else "")
                modifier_round += 1
            used_names.add(product_name)
            cost = round(np.random.uniform(5.0, 60.0), 2)
            rows.append({
                "Product_ID": f"PROD-{len(rows) + 1:04d}",
                "Product_Name": product_name,
                "Category": category,
                "Supplier": f"Supplier_{chr(np.random.randint(ord('A'), ord('J') + 1))}{np.random.randint(1, 6)}",
                "Cost": cost,
                "Selling_Price": round(cost * np.random.uniform(1.2, 1.8), 2),
                "Stock": int(np.random.randint(0, 501)),
                "Reorder_Level": int(np.random.randint(20, 51)),
            })

    products = pd.DataFrame(rows)
    # Force ~10 genuine stockouts for the inventory stockout analysis.
    zero_idx = np.random.choice(n, size=min(10, n), replace=False)
    products.loc[zero_idx, "Stock"] = 0
    logger.info("Product catalogue ready: {} products, {} forced stockouts.", n, len(zero_idx))
    return products


def generate_customers_data(sales_df: pd.DataFrame, n: int = 10000) -> pd.DataFrame:
    """Generate the customer base from the sales spending distribution.

    ``Total_Spent`` is sampled from the empirical distribution of the sales
    ``Total`` column, and ``Visit_Count`` is a random 1-200 value scaled by
    each customer's spend relative to the mean, producing a realistic
    positive correlation between the two columns.

    Parameters
    ----------
    sales_df:
        The generated sales DataFrame (used for the spending distribution).
    n:
        Number of customers to create (default 10,000).

    Returns
    -------
    pandas.DataFrame
        The generated customer base.
    """
    n = int(n)
    fake = _seed_everything(config.RANDOM_SEED + 2)
    logger.info("Generating {:,} customers ...", n)

    join_start = date(2020, 1, 1)
    join_end = date(2023, 12, 31)

    total_spent = (
        sales_df["Total"].sample(n=n, replace=True, random_state=config.RANDOM_SEED).to_numpy()
        if sales_df is not None and len(sales_df) > 0
        else np.round(np.random.uniform(20.0, 900.0, size=n), 2)
    )
    spend_ratio = total_spent / total_spent.mean()
    visit_count = np.clip(
        np.round(np.random.uniform(1.0, 200.0, size=n) * spend_ratio), 1, 200
    ).astype(int)

    customers = pd.DataFrame({
        "Customer_ID": [f"CUST-{i:05d}" for i in range(1, n + 1)],
        "Gender": np.random.choice(config.GENDERS, size=n, p=[0.5, 0.5]),
        "Age": np.clip(np.random.normal(35, 12, size=n), 18, 70).astype(int),
        "City": np.random.choice(sorted(set(config.BRANCHES.values())), size=n),
        "Membership": np.random.choice(
            config.MEMBERSHIP_TIERS, size=n, p=[0.10, 0.20, 0.30, 0.40]
        ),
        "Join_Date": [fake.date_between(start_date=join_start, end_date=join_end).isoformat() for _ in range(n)],
        "Total_Spent": np.round(total_spent, 2),
        "Visit_Count": visit_count,
    })
    logger.info("Customer base ready: {:,} customers.", n)
    return customers


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------
def load_to_database(
    sales_df: Optional[pd.DataFrame] = None,
    products_df: Optional[pd.DataFrame] = None,
    customers_df: Optional[pd.DataFrame] = None,
    db_path: Optional[str] = None,
) -> Dict[str, int]:
    """Load the three datasets into SQLite using the schema.sql DDL.

    The helper column ``Is_Injected_Anomaly`` is intentionally excluded from
    the database so the tables match schema.sql exactly (the flag lives in
    the CSV, which is what the anomaly-detection module reads).

    Tables are dropped and re-created through schema.sql before an append:
    this gives the same reset semantics as ``if_exists='replace'`` while
    preserving the declared PRIMARY KEYs, CHECKs and the four analytical
    indexes (pandas' native 'replace' would rebuild the tables without
    them).

    Any DataFrame left as ``None`` is read from the CSVs in
    ``data/augmented/`` so the database can be rebuilt without regenerating.

    Returns
    -------
    dict
        Row counts per table, e.g. ``{"sales": 500000, ...}``.
    """
    from sqlalchemy import create_engine, text

    db_path = Path(db_path) if db_path else Path(config.DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if sales_df is None:
        sales_df = pd.read_csv(config.SALES_DATA_FILE)
        logger.info("Loaded {:,} sales rows from {} for DB import.", len(sales_df), config.SALES_DATA_FILE)
    if products_df is None:
        products_df = pd.read_csv(config.PRODUCTS_DATA_FILE)
    if customers_df is None:
        # keep_default_na=False: the literal membership tier "None" must not
        # be parsed as NaN (it is in pandas' default NA list).
        customers_df = pd.read_csv(config.CUSTOMERS_DATA_FILE, keep_default_na=False)
        customers_df["Membership"] = customers_df["Membership"].fillna("None")

    schema_path = Path(config.PROJECT_ROOT) / "database" / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")

    logger.info("Rebuilding SQLite database at {} ...", db_path)
    with sqlite3.connect(str(db_path)) as conn:
        for table in ("sales", "products", "customers"):
            conn.execute(f"DROP TABLE IF EXISTS {table};")
        conn.executescript(schema_sql)

    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    try:
        sales_db = sales_df.drop(columns=["Is_Injected_Anomaly"], errors="ignore")
        sales_db.to_sql("sales", engine, if_exists="append", index=False, chunksize=50_000)
        products_df.to_sql("products", engine, if_exists="append", index=False)
        customers_df.to_sql("customers", engine, if_exists="append", index=False)

        row_counts = {}
        with engine.connect() as conn:
            for table in ("sales", "products", "customers"):
                row_counts[table] = int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar())
                logger.info("Table '{}': {:,} rows loaded.", table, row_counts[table])
    finally:
        engine.dispose()

    logger.info("Database load complete: {}", row_counts)
    return row_counts


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def generate_all(n_records: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full generation pipeline and persist every artefact.

    1. Generate sales / products / customers datasets.
    2. Save them as CSVs under ``data/augmented/``.
    3. Load everything into the SQLite database.

    Returns
    -------
    tuple
        The three generated DataFrames (sales, products, customers).
    """
    config.ensure_directories()
    logger.info("=== Data generation started ===")

    sales = generate_sales_data(n_records or config.NUM_RECORDS)
    products = generate_products_data(200)
    customers = generate_customers_data(sales, n=10000)

    sales.to_csv(config.SALES_DATA_FILE, index=False)
    logger.info("Saved sales -> {}", config.SALES_DATA_FILE)
    products.to_csv(config.PRODUCTS_DATA_FILE, index=False)
    logger.info("Saved products -> {}", config.PRODUCTS_DATA_FILE)
    customers.to_csv(config.CUSTOMERS_DATA_FILE, index=False)
    logger.info("Saved customers -> {}", config.CUSTOMERS_DATA_FILE)

    load_to_database(sales, products, customers)

    logger.info(
        "=== Data generation complete: {:,} sales / {} products / {:,} customers ===",
        len(sales), len(products), len(customers),
    )
    return sales, products, customers


if __name__ == "__main__":
    generate_all()
