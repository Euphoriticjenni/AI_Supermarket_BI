# 🛒 AI-Powered Supermarket Analytics & BI System

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-success)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)

A complete, end-to-end Business Intelligence system for supermarket retail
operations — from synthetic data generation to an interactive dashboard, an
offline natural-language assistant, and automated executive PDF reports.

---

## 📋 Overview

**AI_Supermarket_BI** is a Python-based Business Intelligence platform that
simulates and analyses the full lifecycle of supermarket retail data. The
pipeline starts by synthesising a realistic two-year transaction history —
**500,000 sales lines** across 3 branches (Yangon, Mandalay, Naypyitaw),
6 product lines and 3 payment methods, plus a 10,000-customer base and a
200-product catalogue — then validates every record with schema contracts,
cleans it (duplicates, nulls, outliers, dtypes) and enriches it into a
**37-column feature table** (temporal, financial, rolling, lag and basket
features) that lands in both Parquet and a SQLite warehouse.

On top of that foundation sits a lightweight analytics layer built entirely
on **pandas / NumPy / Plotly / SQL**: rule-based **RFM customer segmentation**
with CLV scoring, daily/weekly/monthly **trend analysis** with moving
averages and a seasonal-adjusted 30-day revenue projection, and an
**inventory engine** that computes safety stock, reorder points and ABC
classification. Twelve focused Python modules in `src/` keep generation,
validation, analytics, visualisation and reporting cleanly separated, and a
16-function chart factory guarantees a consistent look across every view.

Everything is surfaced through a **6-view Streamlit dashboard** (overview,
sales, customers, inventory, trends, assistant), an **offline AI assistant**
that turns plain-English questions into SQL and answers with text, tables and
charts — no API key, no LLM required — and an **automated multi-page
executive PDF report** rendered with matplotlib + fpdf2. The whole system is
covered by a pytest suite, shipped as a Docker image, and linted/tested on
GitHub Actions (Python 3.10 & 3.11).

## 🎯 Problem Statement

Supermarkets generate a flood of point-of-sale data, but the people who run
them rarely have direct access to it:

- **Managers** answer basic questions ("Which branch made more money last
  quarter?") by exporting spreadsheets or waiting on analysts.
- **Inventory decisions** are guesswork — without demand statistics and
  reorder points, stores both stock out on best-sellers and overstock the
  tail.
- **Customers are invisible** — the business cannot tell its champions from
  its churned customers without segmentation work.
- **Commercial BI tools** are expensive, heavyweight, and often require cloud
  infrastructure and ML platforms that a single analyst cannot maintain.

This project demonstrates that a **complete, self-serve BI stack** —
dashboard, natural-language querying, segmentation, inventory planning and
executive reporting — can run **100% offline on one machine**, using nothing
but deterministic rules, SQL and lightweight statistics. It is fully
reproducible (seeded, env-configurable) and designed so that the synthetic
data layer can later be swapped for a real POS feed without touching the
analytics or presentation layers.

## ✨ Key Features

- 🖥️ **Interactive dashboard** — 6 Streamlit views with shared filters (date range, branch, city, product line), headline KPIs and ~15 Plotly charts
- 👥 **Customer analytics** — rule-based RFM segmentation (6 segments), deterministic proxy customers, CLV scoring with Low/Medium/High tiers
- 📈 **Trends & forecasting** — daily/weekly/monthly aggregation, 7/30-day moving averages, year-over-year comparison, seasonal heatmap and a seasonal-adjusted 30-day projection with confidence band
- 📦 **Inventory management** — safety stock (Z·σ·√L), reorder points, prioritised reorder report with stock-action flags, and ABC product-line classification
- 🤖 **AI assistant** — offline regex→SQL engine (27 query templates) answering plain-English questions with text, tables and charts; zero API keys
- 📄 **Automated PDF reports** — multi-page executive report (revenue, profit, branches, product lines, ratings) built with matplotlib + fpdf2 at the click of a button
- ✅ **Tested** — pytest suite covering preprocessing, analytics, inventory and data validation contracts
- 🐳 **Dockerised** — one-command build/run with volume-persisted data, database, reports and logs, plus a Streamlit healthcheck
- 🔄 **CI/CD** — GitHub Actions pipeline linting (flake8) and testing (pytest) on a Python 3.10/3.11 matrix
- 🅰️ **ABC analysis** — cumulative-revenue classification (A/B/C) shared by the analytics and inventory modules

## 🏗️ Architecture

```
 ┌─────────────────────────────── DATA LAYER ────────────────────────────────┐
 │                                                                           │
 │   src/data_generator ──────► data/augmented/        data/raw/             │
 │   (seeded Faker, 500K rows)   supermarket_sales.csv   (landing zone)      │
 │                               customers.csv                               │
 │                               products.csv                                │
 │                                        │                                  │
 │                                        ▼                                  │
 │   src/data_validation ──► pandera schema contracts (fail fast)            │
 │   src/preprocessing ────► dedupe, nulls, outliers ──► sales_processed     │
 │   src/feature_engineering ► temporal / financial / rolling / lag /        │
 │                              basket features ──► sales_features.parquet   │
 │                                                   (471,588 × 37)          │
 │                                        │                                  │
 │                                        ▼                                  │
 │   SQLite warehouse (database/supermarket.db)  ◄── sales / customers /     │
 │                                                   products tables         │
 └────────────────────────────────┬──────────────────────────────────────────┘
                                  │
 ┌────────────────────────────────▼──────────── ANALYTICS LAYER ─────────────┐
 │                                                                           │
 │   src/analytics ──── RFM + CLV · product & ABC analysis · trends ·        │
 │                      moving averages · YoY · seasonal projection          │
 │   src/inventory ──── safety stock · reorder points · reorder report ·     │
 │                      ABC classification                                   │
 │   src/visualization  16-function Plotly chart factory (shared palette)    │
 │   src/logger ─────── Loguru structured logging across every module        │
 └────────────────────────────────┬──────────────────────────────────────────┘
                                  │
 ┌────────────────────────────────▼───────── PRESENTATION LAYER ─────────────┐
 │                                                                           │
 │   Streamlit dashboard (dashboard/)   AI assistant        Report generator │
 │   ├─ 🛒 Home (KPIs + charts)         src/ai_assistant    src/               │
 │   ├─ 📊 Sales Analytics              regex → SQL →      report_generator   │
 │   ├─ 👥 Customer Analytics           text / table /     fpdf2 + matplotlib │
 │   ├─ 📦 Inventory                    chart (offline)    → reports/*.pdf    │
 │   ├─ 📈 Trends & Forecasting                                               │
 │   └─ 🤖 AI Assistant                                                       │
 └───────────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| Language | Python 3.10 / 3.11 | Core runtime (typed, modular `src/` package) |
| Data processing | Pandas 2.1 | Aggregations, RFM, trends, inventory maths |
| Numerics | NumPy 1.26 | Scoring, statistics, OLS trend fitting |
| Visualisation | Plotly 5.18 | 16-function interactive chart factory |
| Dashboard | Streamlit 1.29 | 6-view multipage BI application |
| Database | SQLite + SQLAlchemy 2.0 | Lightweight warehouse the assistant queries |
| Validation | Pandera 0.18 | Schema contracts on raw/processed data |
| Synthetic data | Faker 21.0 | Realistic customers, products and transactions |
| PDF reports | fpdf2 2.7 + Matplotlib 3.8 | Executive report rendering with static PNG charts |
| Time series | Statsmodels 0.14 | Seasonal decomposition of daily sales |
| Logging | Loguru 0.7 | Structured, rotating application logs |
| Config | python-dotenv | `.env`-driven paths, seeds and record counts |
| Testing | pytest 7.4 | Unit/contract tests for all core modules |
| Container | Docker + docker-compose | One-command deployment with persisted volumes |
| CI/CD | GitHub Actions | flake8 + pytest on Python 3.10/3.11 matrix |
| Database engine | SQLite (file-based) | Zero-admin warehouse for sales/customers/products |

## 📁 Project Structure

```
AI_Supermarket_BI/
├── dashboard/                     # Streamlit multipage application
│   ├── app.py                     #   🛒 Home — KPIs + overview charts
│   └── pages/
│       ├── 1_📊_Sales_Analytics.py    # period aggregation, geography, products
│       ├── 2_👥_Customer_Analytics.py # RFM segments, CLV, demographics
│       ├── 3_📦_Inventory.py          # safety stock, reorder report, ABC
│       ├── 4_📈_Trends.py             # projection, moving averages, YoY
│       └── 5_🤖_AI_Assistant.py       # offline Q&A + executive PDF report
├── src/                           # 12 core Python modules
│   ├── __init__.py                #   package documentation
│   ├── config.py                  #   all paths, seeds, business constants
│   ├── logger.py                  #   Loguru setup
│   ├── data_generator.py          #   seeded synthetic data → CSV + SQLite
│   ├── data_validation.py         #   Pandera schema contracts
│   ├── preprocessing.py           #   cleaning pipeline → sales_processed
│   ├── feature_engineering.py     #   37-column enriched feature table
│   ├── analytics.py               #   RFM/CLV, product & ABC, trends
│   ├── inventory.py               #   safety stock, reorder report, ABC
│   ├── visualization.py           #   16-function Plotly chart factory
│   ├── ai_assistant.py            #   offline regex→SQL business assistant
│   └── report_generator.py        #   executive PDF (fpdf2 + matplotlib)
├── data/
│   ├── raw/                       # landing zone (kept empty in repo)
│   ├── augmented/                 # generated datasets
│   │   ├── supermarket_sales.csv  #   500,000 transaction lines
│   │   ├── customers.csv          #   10,000 customers
│   │   └── products.csv           #   200 products (Stock, Reorder_Level)
│   └── processed/                 # pipeline artefacts (generated)
│       ├── sales_processed.csv/.parquet
│       ├── sales_features.parquet #   471,588 rows × 37 columns
│       ├── abc_analysis.csv
│       ├── reorder_report.csv
│       ├── outliers_removed.csv
│       └── analytics_summary.parquet
├── database/
│   ├── schema.sql                 # warehouse DDL
│   └── supermarket.db             # SQLite warehouse (generated)
├── tests/                         # pytest suites
│   ├── test_preprocessing.py
│   ├── test_analytics.py
│   ├── test_inventory.py
│   └── test_data_validation.py
├── notebooks/
│   └── 01_EDA.ipynb               # exploratory analysis
├── reports/                       # generated executive PDFs
├── logs/                          # Loguru application logs
├── .env.example                   # optional configuration template
├── .github/workflows/ci.yml       # lint + test pipeline (3.10 / 3.11)
├── Dockerfile                     # Streamlit runtime image (port 8501)
├── docker-compose.yml             # volumes for data/db/reports/logs
├── Makefile                       # make generate-data | dashboard | test | …
├── requirements.txt               # pinned dependencies
├── setup.py                       # pip install -e . support (MIT)
└── audit.py                       # project self-audit script
```

## 🚀 Quick Start

**Prerequisites:** Python 3.10+ and Make (or run the underlying commands manually — every `make` target is one line you can copy from the `Makefile`).

```bash
# 1. Clone
git clone https://github.com/your-org/AI_Supermarket_BI.git
cd AI_Supermarket_BI

# 2. Install dependencies (optionally in a virtualenv)
pip install -r requirements.txt

# 3. Build the data pipeline
make generate-data      # seed 500K transactions + customers + products → CSV & SQLite
make preprocess         # validate & clean → data/processed/sales_processed
make features           # enrich to 37 columns → data/processed/sales_features.parquet
make analytics          # RFM + trends + product/ABC summary artefacts

# 4. Launch the dashboard
make dashboard          # → http://localhost:8501
```

Optional extras:

```bash
make test               # pytest suite
make lint               # flake8 (max-line-length=120)
make report             # generate an executive PDF without the dashboard
make status             # inspect what the pipeline has produced
```

**Docker:**

```bash
make docker-build       # build the supermarket-bi image
make docker-run         # start it detached → http://localhost:8501
make docker-stop        # tear down
```

Volumes mount `data/`, `database/`, `reports/` and `logs/`, so container
rebuilds keep their state.

**Configuration (optional):** copy `.env.example` to `.env` to tune
`NUM_RECORDS`, `RANDOM_SEED`, `DATABASE_PATH`, `LOG_LEVEL` and `REPORT_PATH`.
Everything works fully offline; an `OPENAI_API_KEY` is never required.

## 📊 Dashboard Pages

| Page | What it shows |
|---|---|
| 🛒 **Home** | 8 headline KPIs (revenue, profit, orders, AOV, customers, rating, best branch & product line) plus monthly revenue, product-line treemap, daily trend, branch comparison and payment/customer-mix charts. Every chart reacts to the sidebar filters. |
| 📊 **Sales Analytics** | Daily/Weekly/Monthly revenue aggregation selector, branch & city performance, product-line revenue and profit, hourly and weekday patterns, payment & customer-type mix, year-over-year comparison, and the OLS revenue trend line. |
| 👥 **Customer Analytics** | Rule-based RFM segmentation with segment sizes and profiles (avg R/F/M scores, monetary, CLV), CLV distribution and tiers, Top-20 customers by CLV, gender & age demographics, and the Member-vs-Normal transaction breakdown. |
| 📦 **Inventory** | Stock-position metrics, per-product safety stock and reorder points (service level 90/95/99% and supplier lead time adjustable in the sidebar), the prioritised reorder report with action flags, and ABC classification of product lines. |
| 📈 **Trends & Forecasting** | Seasonal-adjusted 30-day projection with confidence band, latest MoM/YoY growth, daily revenue with 7/30-day moving averages, monthly year-over-year lines, a Month×Year seasonal heatmap, weekly revenue and the MoM growth table. |
| 🤖 **AI Assistant** | Offline business Q&A (regex→SQL over the warehouse) with answers, tables and charts, an executive summary with key insights, and a one-click executive PDF report generator with in-browser download. |

All pages share a consistent design: `@st.cache_data` loaders with
parquet-first/CSV-fallback, `layout="wide"`, a common Set2 Plotly palette,
and empty-state warnings when filters exclude everything.

## 🤖 AI Assistant Examples

The assistant matches questions against 27 ordered regex templates, runs the
SQL on the SQLite warehouse, and answers with text, a table and — when it
helps — a chart. Real output from the current dataset:

**Q: "What was total revenue by branch?"**

```sql
SELECT Branch, City, ROUND(SUM(Total),2) as Revenue,
       ROUND(SUM(Gross_Income),2) as Profit,
       COUNT(DISTINCT Invoice_ID) as Orders
FROM sales GROUP BY Branch, City ORDER BY Revenue DESC
```

> Here are the top 3 of 3 results:
> - Branch: A, City: Yangon, Revenue: 34,902,087.62, Profit: 1,662,025.03, Orders: 166,919
> - Branch: B, City: Mandalay, Revenue: 34,829,508.18, Profit: 1,658,569.91, Orders: 166,620
> - Branch: C, City: Naypyitaw, Revenue: 34,643,402.62, Profit: 1,649,708.40, Orders: 166,461
>
> A leads with 34,902,087.62 (258,685.00 more than C).

**Q: "Which products need restocking?"**

```sql
SELECT Product_ID, Product_Name, Stock, Reorder_Level,
       CASE WHEN Stock = 0 THEN 'STOCKOUT'
            WHEN Stock < Reorder_Level THEN 'LOW'
            ELSE 'OK' END as Status
FROM products ORDER BY Stock ASC LIMIT 20
```

> Here are the top 3 of 20 results:
> - Product ID: PROD-0002, Product Name: Herbal Shampoo, Stock: 0, Reorder Level: 47, Status: STOCKOUT
> - Product ID: PROD-0004, Product Name: Vitamin C Serum, Stock: 0, Reorder Level: 20, Status: STOCKOUT
> - Product ID: PROD-0077, Product Name: Storage Box, Stock: 0, Reorder Level: 35, Status: STOCKOUT

**Q: "What is the basket size (AOV)?"**

```sql
SELECT ROUND(AVG(Total),2) as Avg_Order_Value, ROUND(MIN(Total),2) as Min_Order,
       ROUND(MAX(Total),2) as Max_Order,
       ROUND(SUM(Total)/COUNT(DISTINCT Invoice_ID),2) as Revenue_Per_Order
FROM sales
```

> Here's the summary: Avg Order Value = 208.75, Min Order = 10.50,
> Max Order = 46,579.05, Revenue Per Order = 208.75.

The assistant never raises: unmatched or failing questions return a helpful
message instead of an error, and it works identically offline.

## 📈 Analytics Capabilities

| Capability | Method | Where |
|---|---|---|
| RFM segmentation (6 segments) | Rule-based scoring on Recency / Frequency / Monetary (1–5 scales) | `src/analytics.py` → Customers page |
| Customer lifetime value | AOV × frequency × annualised recency factor, tiered Low/Medium/High | `src/analytics.py` → Customers page |
| Daily / weekly / monthly trends | `groupby` aggregation with ISO weeks and Year-Month keys | `src/analytics.py` → Trends page |
| Moving averages | 7-day and 30-day rolling means on daily revenue | `src/analytics.py` → Trends page |
| Year-over-year comparison | Month×Year pivot and overlaid monthly lines | `src/analytics.py` → Trends page |
| 30-day revenue projection | Seasonal-adjusted moving average with ±1σ band (√30 scaling) | `src/analytics.py` → Trends page |
| Seasonal decomposition | Additive `seasonal_decompose` (period = 30 days) | `src/visualization.py` |
| ABC classification | Cumulative revenue share (A ≤ 80%, B ≤ 95%, C remainder) | `src/inventory.py` / `src/analytics.py` |
| Safety stock & reorder points | Z·σ·√L with service levels 90/95/99% and lead-time buffer | `src/inventory.py` → Inventory page |
| Reorder prioritisation | Action flags (urgent → overstock) sorted by headroom | `src/inventory.py` → Inventory page |
| Correlation analysis | Numeric-feature correlation matrix (price, quantity, revenue, rating…) | `src/visualization.py` |
| Text-to-SQL Q&A | 27 ordered regex→SQL templates over the SQLite warehouse | `src/ai_assistant.py` → Assistant page |
| Executive reporting | Matplotlib charts + fpdf2 multi-page PDF layout | `src/report_generator.py` |

## ⚠️ Known Limitations

- **Synthetic data.** All transactions, customers and products are generated
  (seeded Faker, env-tunable record count). Distributions are realistic but
  the 50 injected extreme transactions exist specifically to exercise the
  outlier pipeline — absolute KPI values are not market truths.
- **No machine learning.** By design, every number comes from deterministic
  rules and statistics — segmentation, CLV and the 30-day projection are
  transparent heuristics, not trained models.
- **SQLite only.** The warehouse is a single-file database: fine for a
  single-analyst dashboard, but there is no concurrency, user management or
  network access.
- **Proxy customers.** The sales table carries no customer key, so RFM/CLV
  use a deterministic invoice-ID proxy (`crc32`) — "customers" are
  invoice clusters, not loyalty-card holders.
- **Single-user dashboard.** Streamlit runs without authentication, HTTPS or
  row-level security; it is not exposed for multi-user production use as-is.
- **PDF statics via matplotlib.** Plotly figures are interactive in the
  browser, but static export deliberately routes through matplotlib because
  kaleido hangs on some Windows hosts.

## 🔮 Future Enhancements

- 🗄️ Swap SQLite for PostgreSQL / a cloud warehouse behind the same SQL layer
- 🔐 Add authentication and per-user dashboard scopes for real deployments
- 🔄 Scheduled pipeline refresh (cron / Airflow) feeding a live POS ingest
- 🔔 Reorder alerting: threshold emails or Slack notifications from the reorder report
- 📊 Excel / Google Sheets exports of every dashboard table
- 🧠 Optional statistical demand forecasting to replace the heuristic projection once real data arrives
- 🌐 i18n and localised currency formatting for multi-country rollouts
- ☸️ Kubernetes manifests and horizontal Streamlit scaling behind a gateway

## 📄 License

Released under the **MIT License** — free to use, modify and distribute;
see the `license="MIT"` declaration in `setup.py` and the standard MIT
grant terms.
