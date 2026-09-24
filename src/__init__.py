"""AI_Supermarket_BI source package.

This package contains every building block of the AI-Powered Supermarket
Analytics & Business Intelligence System:

- data_generator: synthetic data generation and SQLite loading
- data_validation: pandera-based schema enforcement
- preprocessing: cleaning pipeline (duplicates, nulls, outliers, dtypes)
- feature_engineering: temporal, financial, rolling and basket features
- visualization: plotly/matplotlib chart factory used by the dashboard
- analytics: RFM segments, product/ABC analysis, trends and projections
- inventory: stock status, safety stock, reorder points and ABC analysis
- ai_assistant: offline text-to-SQL business assistant (no API key)
- report_generator: automated executive PDF reports (fpdf2 + matplotlib)

Design constraints: pure pandas/numpy/plotly/SQL — no machine-learning
libraries anywhere in this package — and no kaleido / plotly.to_image
static export (it hangs on some Windows hosts); PNG output goes through
matplotlib instead.
"""
