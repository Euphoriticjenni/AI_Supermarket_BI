-- ============================================================
-- AI_Supermarket_BI - SQLite schema
-- Three tables: sales, products, customers
-- Executed by src/data_generator.py::load_to_database()
-- ============================================================

CREATE TABLE IF NOT EXISTS sales (
    Invoice_ID        TEXT PRIMARY KEY,
    Date              DATE NOT NULL,
    Time              TIME NOT NULL,
    Branch            TEXT NOT NULL,
    City              TEXT NOT NULL,
    Customer_Type     TEXT NOT NULL,
    Gender            TEXT NOT NULL,
    Product_Line      TEXT NOT NULL,
    Unit_Price        REAL NOT NULL,
    Quantity          INTEGER NOT NULL,
    Tax_5pct          REAL NOT NULL,
    Total             REAL NOT NULL,
    Payment           TEXT NOT NULL,
    COGS              REAL NOT NULL,
    Gross_Margin_Pct  REAL NOT NULL,
    Gross_Income      REAL NOT NULL,
    Rating            REAL,
    CHECK (Unit_Price > 0),
    CHECK (Quantity > 0),
    CHECK (Tax_5pct >= 0),
    CHECK (Total > 0),
    CHECK (COGS > 0),
    CHECK (Rating IS NULL OR (Rating >= 1 AND Rating <= 10))
);

CREATE TABLE IF NOT EXISTS products (
    Product_ID     TEXT PRIMARY KEY,
    Product_Name   TEXT NOT NULL,
    Category       TEXT NOT NULL,
    Supplier       TEXT NOT NULL,
    Cost           REAL NOT NULL,
    Selling_Price  REAL NOT NULL,
    Stock          INTEGER NOT NULL,
    Reorder_Level  INTEGER NOT NULL,
    CHECK (Cost > 0),
    CHECK (Selling_Price > 0),
    CHECK (Stock >= 0),
    CHECK (Reorder_Level >= 0)
);

CREATE TABLE IF NOT EXISTS customers (
    Customer_ID   TEXT PRIMARY KEY,
    Gender        TEXT NOT NULL,
    Age           INTEGER NOT NULL,
    City          TEXT NOT NULL,
    Membership    TEXT NOT NULL,
    Join_Date     DATE NOT NULL,
    Total_Spent   REAL DEFAULT 0,
    Visit_Count   INTEGER DEFAULT 0,
    CHECK (Age >= 18),
    CHECK (Total_Spent >= 0),
    CHECK (Visit_Count >= 0)
);

-- ------------------------------------------------------------
-- Analytical indexes (used heavily by the AI assistant and
-- the Streamlit dashboard aggregation queries).
-- ------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_sales_date         ON sales (Date);
CREATE INDEX IF NOT EXISTS idx_sales_branch       ON sales (Branch);
CREATE INDEX IF NOT EXISTS idx_sales_product_line ON sales (Product_Line);
CREATE INDEX IF NOT EXISTS idx_sales_city         ON sales (City);
