"""Generate realistic, reproducible NexusBI supermarket datasets.

The generator deliberately keeps the expensive sales work vectorised.  Faker
is used only for customer names and email addresses; dates, quantities,
prices, customer assignment, and anomaly injection are all produced with
NumPy.  Run the complete pipeline with::

    python -c "from src.data_generator import generate_all; generate_all()"
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

DATE_START = pd.Timestamp("2024-01-01")
DATE_END = pd.Timestamp("2025-12-31")

BRANCHES = [
    "Downtown Central",
    "Northgate Mall",
    "Westside Plaza",
    "East End Market",
    "Southpark Express",
]
PAYMENT_METHODS = [
    "Credit Card",
    "Debit Card",
    "Cash",
    "UPI/Mobile",
    "Store Card",
    "Gift Card",
]
PAYMENT_PROBABILITIES = np.array([0.35, 0.25, 0.15, 0.18, 0.05, 0.02])
MEMBERSHIP_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
MEMBERSHIP_PROBABILITIES = np.array([0.60, 0.25, 0.10, 0.05])
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_PROBABILITIES = np.array([0.10, 0.30, 0.30, 0.15, 0.10, 0.05])

# The first price in each tuple is the typical retail price.  The catalogue is
# expanded with five pack/quality variants, giving 600 products by default.
PRODUCT_TEMPLATES: dict[str, list[tuple[str, str, float, int, bool, bool]]] = {
    "Fresh Produce": [
        ("Bananas", "Fruit", 0.59, 7, True, False),
        ("Gala Apples", "Fruit", 1.99, 21, True, False),
        ("Tomatoes", "Vegetables", 2.49, 10, True, False),
        ("Romaine Lettuce", "Leafy Greens", 2.29, 7, True, False),
        ("Yellow Onions", "Vegetables", 1.49, 30, True, False),
        ("Russet Potatoes", "Vegetables", 1.29, 45, True, False),
        ("Baby Carrots", "Vegetables", 1.79, 14, True, False),
        ("Red Grapes", "Fruit", 3.99, 14, True, False),
        ("Navel Oranges", "Fruit", 2.49, 21, True, False),
        ("Broccoli Crowns", "Vegetables", 2.99, 10, True, False),
        ("Baby Spinach", "Leafy Greens", 3.49, 8, True, True),
        ("Hass Avocados", "Fruit", 1.25, 10, True, False),
        ("Ataulfo Mangoes", "Fruit", 1.49, 10, True, False),
        ("Strawberries", "Berries", 3.99, 5, True, False),
    ],
    "Dairy & Eggs": [
        ("Whole Milk", "Milk", 3.49, 14, True, False),
        ("Skim Milk", "Milk", 3.39, 14, True, False),
        ("Almond Milk", "Milk Alternatives", 4.29, 30, True, True),
        ("Cheddar Cheese", "Cheese", 5.99, 60, True, False),
        ("Mozzarella Cheese", "Cheese", 5.49, 45, True, False),
        ("Greek Yogurt", "Yogurt", 1.39, 30, True, False),
        ("Salted Butter", "Butter", 4.49, 60, True, False),
        ("Large Grade A Eggs", "Eggs", 3.99, 30, True, False),
        ("Cream Cheese", "Cheese", 3.29, 45, True, False),
        ("Sour Cream", "Dairy", 2.49, 30, True, False),
    ],
    "Meat & Seafood": [
        ("Chicken Breast", "Poultry", 6.99, 5, True, False),
        ("Ground Beef", "Beef", 7.99, 5, True, False),
        ("Salmon Fillet", "Seafood", 12.99, 3, True, False),
        ("Jumbo Shrimp", "Seafood", 10.99, 3, True, False),
        ("Pork Chops", "Pork", 8.49, 5, True, False),
        ("Lamb Shoulder", "Lamb", 11.99, 5, True, False),
        ("Turkey Breast", "Poultry", 8.99, 5, True, False),
        ("Tuna Steak", "Seafood", 13.49, 3, True, False),
        ("Center Cut Bacon", "Pork", 6.49, 21, True, False),
        ("Italian Sausages", "Prepared Meat", 5.99, 14, True, False),
    ],
    "Bakery": [
        ("White Bread", "Bread", 2.49, 7, True, False),
        ("Whole Wheat Bread", "Bread", 3.49, 7, True, True),
        ("Butter Croissants", "Pastries", 4.99, 4, True, False),
        ("Plain Bagels", "Bread", 3.99, 7, True, False),
        ("Blueberry Muffins", "Pastries", 4.49, 5, True, False),
        ("Chocolate Cake", "Cakes", 12.99, 4, True, False),
        ("Chocolate Chip Cookies", "Cookies", 3.99, 21, False, False),
        ("Glazed Donuts", "Pastries", 5.49, 3, True, False),
        ("Pita Bread", "Bread", 3.29, 10, True, False),
        ("Garlic Naan", "Bread", 4.49, 10, True, False),
    ],
    "Beverages": [
        ("Cola", "Soft Drinks", 2.49, 180, False, False),
        ("Diet Cola", "Soft Drinks", 2.49, 180, False, False),
        ("Orange Juice", "Juice", 4.99, 21, True, False),
        ("Apple Juice", "Juice", 4.49, 30, False, False),
        ("Bottled Water", "Water", 4.99, 365, False, False),
        ("Sparkling Water", "Water", 5.49, 365, False, False),
        ("Energy Drink", "Energy Drinks", 2.99, 270, False, False),
        ("Iced Tea", "Ready-to-Drink", 2.49, 180, False, False),
        ("Coconut Water", "Juice", 3.49, 120, False, True),
        ("Lemonade", "Juice", 3.99, 30, True, False),
    ],
    "Snacks": [
        ("Potato Chips", "Chips", 3.49, 120, False, False),
        ("Tortilla Chips", "Chips", 3.99, 120, False, False),
        ("Pretzels", "Savoury Snacks", 2.99, 150, False, False),
        ("Microwave Popcorn", "Popcorn", 4.49, 180, False, False),
        ("Granola Bars", "Snack Bars", 4.99, 180, False, True),
        ("Trail Mix", "Nuts & Seeds", 6.99, 120, False, True),
        ("Saltine Crackers", "Crackers", 2.49, 180, False, False),
        ("Rice Cakes", "Healthy Snacks", 3.49, 120, False, False),
        ("Mixed Nuts", "Nuts & Seeds", 8.99, 180, False, True),
        ("Chocolate Candy Bars", "Confectionery", 1.49, 240, False, False),
    ],
    "Frozen Foods": [
        ("Frozen Pizza", "Frozen Meals", 6.99, 365, False, False),
        ("Vanilla Ice Cream", "Ice Cream", 5.99, 365, True, False),
        ("Frozen Vegetables", "Frozen Produce", 2.99, 365, False, True),
        ("Chicken Nuggets", "Frozen Poultry", 7.49, 365, False, False),
        ("Frozen Fries", "Frozen Sides", 3.99, 365, False, False),
        ("Fish Sticks", "Frozen Seafood", 6.49, 365, False, False),
        ("Frozen Dumplings", "Frozen Meals", 5.49, 365, False, False),
        ("Mango Sorbet", "Frozen Desserts", 5.49, 365, True, False),
        ("Frozen Waffles", "Frozen Breakfast", 4.49, 365, False, False),
        ("Frozen Berry Blend", "Frozen Produce", 6.99, 365, False, True),
    ],
    "Pantry Staples": [
        ("Long Grain Rice", "Rice & Grains", 4.99, 730, False, False),
        ("Spaghetti Pasta", "Pasta", 2.29, 730, False, False),
        ("All-Purpose Flour", "Baking", 3.49, 365, False, False),
        ("Granulated Sugar", "Baking", 2.99, 730, False, False),
        ("Sea Salt", "Seasonings", 1.99, 1095, False, False),
        ("Extra Virgin Olive Oil", "Cooking Oils", 9.99, 730, False, True),
        ("Soy Sauce", "Sauces", 3.49, 730, False, False),
        ("Canned Tomatoes", "Canned Goods", 1.79, 730, False, False),
        ("Creamy Peanut Butter", "Spreads", 4.49, 365, False, False),
        ("Corn Flakes Cereal", "Breakfast", 5.49, 365, False, False),
        ("Old Fashioned Oatmeal", "Breakfast", 4.29, 365, False, True),
        ("Wildflower Honey", "Spreads", 7.99, 730, False, True),
    ],
    "Household": [
        ("Paper Towels", "Paper Products", 9.99, 1095, False, False),
        ("Toilet Paper", "Paper Products", 11.99, 1095, False, False),
        ("Dish Soap", "Cleaning", 3.49, 730, False, False),
        ("Laundry Detergent", "Laundry", 12.99, 730, False, False),
        ("Trash Bags", "Storage & Waste", 8.49, 730, False, False),
        ("Kitchen Sponges", "Cleaning", 3.99, 730, False, False),
        ("Hand Soap", "Cleaning", 2.99, 730, False, False),
        ("Liquid Bleach", "Cleaning", 4.99, 365, False, False),
        ("Air Freshener", "Home Fragrance", 3.99, 730, False, False),
        ("Aluminum Foil", "Kitchen Supplies", 5.99, 1095, False, False),
    ],
    "Personal Care": [
        ("Moisturizing Shampoo", "Hair Care", 7.99, 730, False, False),
        ("Fluoride Toothpaste", "Oral Care", 4.49, 730, False, False),
        ("Antiperspirant Deodorant", "Body Care", 5.99, 730, False, False),
        ("Moisturizing Body Wash", "Body Care", 6.49, 730, False, False),
        ("Broad Spectrum Sunscreen", "Skin Care", 10.99, 730, False, False),
        ("Nourishing Lip Balm", "Skin Care", 2.49, 730, False, True),
        ("Hand Cream", "Skin Care", 5.49, 730, False, False),
        ("Disposable Razors", "Shaving", 8.99, 730, False, False),
        ("Hair Conditioner", "Hair Care", 7.99, 730, False, False),
        ("Cotton Swabs", "Personal Hygiene", 2.99, 1095, False, False),
    ],
    "Baby & Kids": [
        ("Newborn Diapers", "Diapers", 11.99, 730, False, False),
        ("Stage 3 Diapers", "Diapers", 14.99, 730, False, False),
        ("Infant Formula", "Baby Nutrition", 24.99, 365, False, False),
        ("Organic Baby Food", "Baby Nutrition", 2.49, 365, False, True),
        ("Sensitive Baby Wipes", "Baby Care", 5.99, 730, False, False),
        ("Children's Vitamins", "Kids Wellness", 9.99, 730, False, False),
        ("Gentle Baby Shampoo", "Baby Care", 6.99, 730, False, False),
        ("Fruit Snack Pouches", "Kids Snacks", 4.99, 180, False, False),
        ("Sippy Cup", "Feeding", 7.99, 1095, False, False),
        ("Kids Toothbrush", "Oral Care", 3.49, 730, False, False),
    ],
    "Health & Wellness": [
        ("Daily Multivitamins", "Vitamins", 12.99, 730, False, False),
        ("Ibuprofen Tablets", "Pain Relief", 8.99, 730, False, False),
        ("Adhesive Bandages", "First Aid", 4.99, 1095, False, False),
        ("Honey Cough Syrup", "Cold & Flu", 9.99, 730, False, True),
        ("Whey Protein Powder", "Sports Nutrition", 29.99, 730, False, False),
        ("Daily Probiotics", "Digestive Health", 19.99, 365, False, False),
        ("Vitamin D3", "Vitamins", 10.99, 730, False, False),
        ("Electrolyte Powder", "Sports Nutrition", 14.99, 730, False, False),
        ("Digital Thermometer", "First Aid", 12.49, 1095, False, False),
        ("Magnesium Capsules", "Vitamins", 13.99, 730, False, False),
    ],
}

CATEGORY_ORDER = list(PRODUCT_TEMPLATES)
CATEGORY_WEIGHTS = np.array([0.16, 0.10, 0.10, 0.08, 0.10, 0.08, 0.07, 0.12, 0.06, 0.05, 0.03, 0.05])
CATEGORY_WEIGHTS /= CATEGORY_WEIGHTS.sum()
VARIANT_NAMES = ["Classic", "Value", "Premium", "Organic", "Family Pack"]
VARIANT_FACTORS = np.array([1.00, 0.90, 1.12, 1.18, 1.30])


def _rng(seed: int) -> np.random.Generator:
    """Return an isolated random generator for reproducible independent calls."""
    return np.random.default_rng(seed)


def generate_products(n_products: int = 600, seed: int = 42) -> pd.DataFrame:
    """Create a varied product catalogue across all twelve categories."""
    if n_products < len(CATEGORY_ORDER):
        raise ValueError(f"n_products must be at least {len(CATEGORY_ORDER)}")

    rng = _rng(seed)
    rows: list[dict[str, object]] = []
    for product_number in range(n_products):
        category = CATEGORY_ORDER[product_number % len(CATEGORY_ORDER)]
        cycle_number = product_number // len(CATEGORY_ORDER)
        category_items = PRODUCT_TEMPLATES[category]
        name, subcategory, price, shelf_life, perishable, organic = category_items[
            cycle_number % len(category_items)
        ]
        # Cycling through product templates and variants keeps the catalogue
        # balanced without making either names or prices flat.
        variant_number = (cycle_number // len(category_items)) % len(VARIANT_NAMES)
        variant = VARIANT_NAMES[variant_number]
        if variant == "Classic":
            display_name = name
        else:
            display_name = f"{variant} {name}"
        if variant == "Family Pack":
            display_price = price * 1.30
        else:
            display_price = price * VARIANT_FACTORS[variant_number]
        display_price *= rng.uniform(0.96, 1.04)
        display_price = max(0.49, round(display_price, 2))
        margin = rng.uniform(0.58, 0.78)
        rows.append({
            "product_id": f"PROD-{product_number + 1:05d}",
            "name": display_name,
            "category": category,
            "subcategory": subcategory,
            "unit_cost": round(display_price * margin, 2),
            "unit_price": display_price,
            "supplier": f"Supplier {chr(65 + (product_number % 12))}-{1 + product_number % 8}",
            "shelf_life_days": shelf_life,
            "is_perishable": bool(perishable),
            "is_organic": bool(organic or variant == "Organic"),
        })

    products = pd.DataFrame(rows)
    logger.info("Generated {:,} products across {} categories.", len(products), products["category"].nunique())
    return products


def generate_customers(n_customers: int = 15_000, seed: int = 42) -> pd.DataFrame:
    """Create customer profiles with stable preferences and realistic segments."""
    if n_customers <= 0:
        raise ValueError("n_customers must be positive")
    rng = _rng(seed + 1)
    fake = Faker()
    fake.seed_instance(seed + 1)

    first_names = np.array([fake.first_name() for _ in range(n_customers)], dtype=object)
    last_names = np.array([fake.last_name() for _ in range(n_customers)], dtype=object)
    email_addresses = []
    for i in range(n_customers):
        raw_email = fake.email()
        domain = raw_email.split("@", 1)[-1]
        local = f"{first_names[i].lower().replace(' ', '.')}.{last_names[i].lower().replace(' ', '.')}"
        email_addresses.append(f"{local}.{i + 1}@{domain}")

    tiers = rng.choice(MEMBERSHIP_TIERS, size=n_customers, p=MEMBERSHIP_PROBABILITIES)
    age_groups = rng.choice(AGE_GROUPS, size=n_customers, p=AGE_PROBABILITIES)
    preferred_payment = rng.choice(PAYMENT_METHODS, size=n_customers, p=PAYMENT_PROBABILITIES)
    preferred_branch = rng.choice(BRANCHES, size=n_customers, p=[0.33, 0.23, 0.20, 0.14, 0.10])
    tier_visit_factor = pd.Series(tiers).map({"Bronze": 0.8, "Silver": 1.1, "Gold": 1.6, "Platinum": 2.2}).to_numpy()
    visits = np.clip(rng.gamma(shape=2.2, scale=1.7, size=n_customers) * tier_visit_factor, 0.2, 18.0)
    join_days = rng.integers(0, (pd.Timestamp("2025-12-31") - pd.Timestamp("2021-01-01")).days + 1, n_customers)
    join_dates = pd.Timestamp("2021-01-01") + pd.to_timedelta(join_days, unit="D")
    phone_digits = rng.integers(0, 10, size=(n_customers, 10)).astype(str)
    # Format phone digits after generation; Faker is deliberately not used for
    # numeric fields in this generator.
    phones = np.array([f"+1-{''.join(row[:3])}-{''.join(row[3:6])}-{''.join(row[6:])}" for row in phone_digits], dtype=object)

    customers = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, n_customers + 1)],
        "first_name": first_names,
        "last_name": last_names,
        "email": email_addresses,
        "phone": phones,
        "join_date": join_dates.strftime("%Y-%m-%d"),
        "membership_tier": tiers,
        "age_group": age_groups,
        "preferred_branch": preferred_branch,
        "avg_weekly_visits": np.round(visits, 2),
        "preferred_payment": preferred_payment,
        "is_active": rng.random(n_customers) < 0.85,
    })
    logger.info("Generated {:,} customers.", len(customers))
    return customers


def _date_probabilities() -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return daily probabilities containing the requested seasonality."""
    dates = pd.date_range(DATE_START, DATE_END, freq="D")
    month_factors = {1: 0.85, 2: 0.92, 3: 1.05, 4: 1.05, 5: 1.0, 6: 1.0,
                     7: 1.10, 8: 1.10, 9: 1.08, 10: 1.0, 11: 1.15, 12: 1.35}
    weekday_factors = {0: 1.0, 1: 1.0, 2: 0.95, 3: 1.0, 4: 1.15, 5: 1.30, 6: 1.20}
    # Split the monthly revenue effect between transaction frequency and
    # basket quantity.  This avoids applying (for example) December's 35%
    # uplift twice while still making both dimensions seasonal.
    factors = np.array([np.sqrt(month_factors[d.month]) * weekday_factors[d.dayofweek] for d in dates])
    return dates, factors / factors.sum()


def _hour_probabilities() -> np.ndarray:
    """Hour weights with near-zero overnight traffic and evening peak."""
    weights = np.array([0.002, 0.001, 0.001, 0.002, 0.001, 0.002, 0.003, 0.006,
                        0.060, 0.080, 0.070, 0.070, 0.100, 0.110, 0.070, 0.060,
                        0.070, 0.110, 0.140, 0.140, 0.120, 0.070, 0.050, 0.040])
    return weights / weights.sum()


def _category_scores(branches: np.ndarray, hours: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Build row-wise category probabilities from branch/hour preferences."""
    base = CATEGORY_WEIGHTS.copy()
    branch_adjustments = np.ones((len(BRANCHES), len(CATEGORY_ORDER)))
    # Branch descriptions are reflected in demand, not merely labels.
    branch_adjustments[0, [3, 4, 5]] = [1.35, 1.30, 1.20]
    branch_adjustments[1, [1, 6, 7, 10]] = [1.35, 1.35, 1.20, 1.30]
    branch_adjustments[2, [0, 2, 11]] = [1.35, 1.25, 1.40]
    branch_adjustments[3, [5, 6, 7]] = [1.30, 1.35, 1.30]
    branch_adjustments[4, [3, 4, 5, 6]] = [1.35, 1.40, 1.25, 1.20]
    hour_adjustments = np.ones((24, len(CATEGORY_ORDER)))
    hour_adjustments[8:11, [1, 3, 4]] = [1.35, 1.50, 1.20]
    hour_adjustments[12:15, [4, 5, 3]] = [1.45, 1.35, 1.20]
    hour_adjustments[17:21, [0, 2, 7, 1]] = [1.25, 1.35, 1.30, 1.20]
    hour_adjustments[[21, 22, 23], [4, 5, 6]] = [1.35, 1.35, 1.25]
    trends_2025 = np.ones(len(CATEGORY_ORDER))
    trends_2025[0] = 1.15
    trends_2025[6] = 0.95
    trends_2025[11] = 1.22
    scores = base * branch_adjustments[branches] * hour_adjustments[hours]
    scores *= np.where(years[:, None] == 2025, trends_2025, 1.0)
    return scores / scores.sum(axis=1, keepdims=True)


def generate_transactions(
    products: Optional[pd.DataFrame] = None,
    customers: Optional[pd.DataFrame] = None,
    n_transactions: int = 750_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate sales lines with seasonality, preferences, and 200 anomalies."""
    # A convenient ``generate_transactions(1000)`` form is useful in notebooks.
    if isinstance(products, (int, np.integer)) and customers is None:
        n_transactions, products = int(products), None
    if n_transactions <= 0:
        raise ValueError("n_transactions must be positive")
    products = products if products is not None else generate_products(seed=seed)
    customers = customers if customers is not None else generate_customers(seed=seed)
    rng = _rng(seed + 2)
    n = int(n_transactions)
    logger.info("Generating {:,} transactions for {} -> {} ...", n, DATE_START.date(), DATE_END.date())

    dates, date_probabilities = _date_probabilities()
    date_indices = rng.choice(len(dates), size=n, p=date_probabilities)
    # Ensure the advertised range is visible even in small smoke-test runs.
    if n >= 2:
        date_indices[0] = 0
        date_indices[1] = len(dates) - 1
    date_values = dates.to_numpy()[date_indices]
    date_series = pd.DatetimeIndex(date_values)
    years = date_series.year.to_numpy()
    months = date_series.month.to_numpy()
    season_factors = np.array([{1: 0.85, 2: 0.92, 3: 1.05, 4: 1.05, 5: 1.0, 6: 1.0,
                                7: 1.10, 8: 1.10, 9: 1.08, 10: 1.0, 11: 1.15, 12: 1.35}[month] for month in months])

    customer_probabilities = customers["membership_tier"].map({"Bronze": 1.0, "Silver": 1.25, "Gold": 1.65, "Platinum": 2.2}).to_numpy(dtype=float)
    customer_probabilities /= customer_probabilities.sum()
    customer_indices = rng.choice(len(customers), size=n, p=customer_probabilities)
    customer_rows = customers.iloc[customer_indices].reset_index(drop=True)

    preferred_branch = customer_rows["preferred_branch"].to_numpy(dtype=object)
    branch_lookup = {branch: index for index, branch in enumerate(BRANCHES)}
    branch_indices = pd.Series(preferred_branch).map(branch_lookup).to_numpy(dtype=np.int8)
    random_branch_indices = rng.choice(len(BRANCHES), size=n, p=[0.33, 0.23, 0.20, 0.14, 0.10])
    branch_indices = np.where(rng.random(n) < 0.78, branch_indices, random_branch_indices)
    hours = rng.choice(24, size=n, p=_hour_probabilities())
    category_probabilities = _category_scores(branch_indices, hours, years)
    category_indices = (rng.random(n)[:, None] <= np.cumsum(category_probabilities, axis=1)).argmax(axis=1)

    product_indices = np.empty(n, dtype=np.int32)
    category_values = np.array(CATEGORY_ORDER, dtype=object)
    for category_index, category in enumerate(CATEGORY_ORDER):
        row_mask = category_indices == category_index
        available = np.flatnonzero(products["category"].to_numpy() == category)
        product_weights = products.iloc[available]["unit_price"].to_numpy(dtype=float) ** -0.35
        product_weights /= product_weights.sum()
        product_indices[row_mask] = rng.choice(available, size=int(row_mask.sum()), p=product_weights)

    product_rows = products.iloc[product_indices].reset_index(drop=True)
    branch_basket_factor = np.array([0.90, 1.25, 1.10, 1.35, 0.85])[branch_indices]
    tier_basket_factor = customer_rows["membership_tier"].map({"Bronze": 0.95, "Silver": 1.05, "Gold": 1.20, "Platinum": 1.35}).to_numpy()
    category_trend = np.ones(n)
    category_trend[(years == 2025) & (category_values[category_indices] == "Fresh Produce")] = 1.15
    category_trend[(years == 2025) & (category_values[category_indices] == "Frozen Foods")] = 0.95
    category_trend[(years == 2025) & (category_values[category_indices] == "Health & Wellness")] = 1.22
    quantity = np.maximum(1, np.rint((rng.poisson(1.0 + branch_basket_factor * tier_basket_factor, n) + 1) * np.sqrt(season_factors) * np.where(years == 2025, 1.08, 1.0) * category_trend * rng.uniform(0.86, 1.14, n))).astype(np.int64)
    quantity = np.minimum(quantity, 18)

    promotion = rng.random(n) < 0.24
    inflation = np.where(years == 2025, 1.03, 1.0)
    seasonal_price = 1.0 + (season_factors - 1.0) * 0.04
    unit_price = product_rows["unit_price"].to_numpy(dtype=float) * inflation * seasonal_price * rng.uniform(0.97, 1.03, n)
    unit_price *= np.where(promotion, rng.uniform(0.80, 0.95, n), 1.0)
    unit_price = np.round(unit_price, 2)
    unit_cost = product_rows["unit_cost"].to_numpy(dtype=float)
    total_amount = np.round(unit_price * quantity, 2)
    profit = np.round((unit_price - unit_cost) * quantity, 2)

    payment = customer_rows["preferred_payment"].to_numpy(dtype=object).copy()
    replace_payment = rng.random(n) >= 0.80
    payment[replace_payment] = rng.choice(PAYMENT_METHODS, size=int(replace_payment.sum()), p=PAYMENT_PROBABILITIES)
    transaction_ids = np.char.add("TXN-", np.char.zfill(np.arange(1, n + 1).astype(str), 8)).astype(object)
    minutes = rng.integers(0, 60, n)
    seconds = rng.integers(0, 60, n)
    datetime_values = date_series + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m") + pd.to_timedelta(seconds, unit="s")

    sales = pd.DataFrame({
        "transaction_id": transaction_ids,
        "datetime": datetime_values.strftime("%Y-%m-%d %H:%M:%S"),
        "date": datetime_values.strftime("%Y-%m-%d"),
        "time": datetime_values.strftime("%H:%M:%S"),
        "customer_id": customer_rows["customer_id"].to_numpy(),
        "product_id": product_rows["product_id"].to_numpy(),
        "product_name": product_rows["name"].to_numpy(),
        "category": product_rows["category"].to_numpy(),
        "subcategory": product_rows["subcategory"].to_numpy(),
        "quantity": quantity,
        "unit_price": unit_price,
        "unit_cost": unit_cost,
        "total_amount": total_amount,
        "profit": profit,
        "payment_method": payment,
        "branch": np.array(BRANCHES, dtype=object)[branch_indices],
        "cashier_id": np.char.add("CASH-", np.char.zfill(rng.integers(1, 81, n).astype(str), 3)),
        "is_return": np.zeros(n, dtype=bool),
        "is_online_order": rng.random(n) < 0.12,
        "promotion_applied": promotion,
        "membership_tier": customer_rows["membership_tier"].to_numpy(),
        "day_of_week": datetime_values.day_name().to_numpy(),
        "hour_of_day": hours,
    })

    # Inject disjoint, labelled-by-behaviour anomalies.  The rows remain in
    # the main CSV so the audit can discover them without a second sidecar.
    order = rng.permutation(n)
    cursor = 0

    def take(count: int) -> np.ndarray:
        nonlocal cursor
        selected = order[cursor:cursor + min(count, n - cursor)]
        cursor += len(selected)
        return selected

    bulk_rows = take(50)
    return_rows = take(30)
    price_rows = take(20)
    zero_rows = take(40)
    duplicate_rows = take(30)
    suspicious_rows = take(30)
    sales.loc[bulk_rows, "quantity"] = rng.integers(51, 121, len(bulk_rows))
    sales.loc[return_rows, "quantity"] = -rng.integers(1, 4, len(return_rows))
    sales.loc[return_rows, "is_return"] = True
    sales.loc[price_rows, "unit_price"] = np.round(sales.loc[price_rows, "unit_price"].to_numpy() * 10.0, 2)
    sales.loc[zero_rows, "total_amount"] = 0.0
    sales.loc[zero_rows, "profit"] = 0.0
    if len(duplicate_rows):
        source_rows = np.arange(len(duplicate_rows), dtype=int)
        sales.loc[duplicate_rows, "transaction_id"] = sales.loc[source_rows, "transaction_id"].to_numpy()
    suspicious_minutes = np.char.zfill(rng.integers(0, 60, len(suspicious_rows)).astype(str), 2)
    suspicious_seconds = np.char.zfill(rng.integers(0, 60, len(suspicious_rows)).astype(str), 2)
    suspicious_times = np.char.add("03:", np.char.add(suspicious_minutes, np.char.add(":", suspicious_seconds)))
    suspicious_dates = sales.loc[suspicious_rows, "date"].to_numpy(dtype=str)
    suspicious_stamps = np.char.add(suspicious_dates, np.char.add(" ", suspicious_times))
    sales.loc[suspicious_rows, "datetime"] = suspicious_stamps
    sales.loc[suspicious_rows, "time"] = suspicious_times
    sales.loc[suspicious_rows, "hour_of_day"] = 3
    sales.loc[suspicious_rows, "day_of_week"] = pd.to_datetime(suspicious_dates).day_name().to_numpy()

    # Recalculate all derived money fields after bulk, return, and price-error edits.
    sales["total_amount"] = np.round(sales["unit_price"] * sales["quantity"], 2)
    sales.loc[zero_rows, "total_amount"] = 0.0
    sales["profit"] = np.round((sales["unit_price"] - sales["unit_cost"]) * sales["quantity"], 2)
    sales.loc[zero_rows, "profit"] = 0.0
    logger.info("Generated {:,} transaction rows; injected 50 bulk, 30 returns, 20 price, 40 void, 30 duplicate-ID, and 30 03:00 anomalies.", len(sales))
    return sales


def generate_all(seed: int = 42, n_transactions: int = 750_000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate and save products, customers, and sales CSVs."""
    config.ensure_directories()
    products = generate_products(seed=seed)
    customers = generate_customers(seed=seed)
    sales = generate_transactions(products, customers, n_transactions=n_transactions, seed=seed)
    products.to_csv(config.PRODUCTS_DATA_FILE, index=False)
    customers.to_csv(config.CUSTOMERS_DATA_FILE, index=False)
    sales.to_csv(config.SALES_DATA_FILE, index=False)
    print(f"Generated {len(products):,} products -> {config.PRODUCTS_DATA_FILE}")
    print(f"Generated {len(customers):,} customers -> {config.CUSTOMERS_DATA_FILE}")
    print(f"Generated {len(sales):,} transactions -> {config.SALES_DATA_FILE}")
    print(f"Revenue: ${sales['total_amount'].sum():,.2f} | Average line: ${sales['total_amount'].mean():,.2f}")
    print("Categories:", sales["category"].nunique(), "| Branches:", sales["branch"].nunique(), "| Payment methods:", sales["payment_method"].nunique())
    return sales, products, customers


if __name__ == "__main__":
    generate_all()
