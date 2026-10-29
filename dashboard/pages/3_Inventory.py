"""NexusBI Inventory Management."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme
from src.forecasting import DemandForecaster

st.set_page_config(page_title="NexusBI · Inventory", page_icon="◆", layout="wide")
theme.inject_theme()
theme.render_sidebar_nav("Inventory")
theme.render_topbar(["Operations", "Inventory"])
st.title("Inventory Management")
st.caption("Monitor stock health, prioritize replenishment, and plan demand by category.")

@st.cache_data(show_spinner="Loading inventory data")
def load_inventory_data():
    return theme.load_data(), theme.load_products()

sales, products = load_inventory_data()
if sales.empty or products.empty:
    theme.render_empty_state("Inventory requires both sales history and a product catalogue.")
    st.metric("Active SKUs", "—")
    st.plotly_chart(go.Figure(), use_container_width=True)
    st.plotly_chart(go.Figure(), use_container_width=True)
    st.dataframe(products, use_container_width=True)
    st.stop()

product_demand = sales.groupby("product_id").agg(units_sold=("quantity", "sum"), revenue=("total_amount", "sum")).reset_index()
inv = products.merge(product_demand, on="product_id", how="left").fillna({"units_sold": 0, "revenue": 0})
inv["daily_demand"] = inv["units_sold"] / max(sales["date"].dt.normalize().nunique(), 1)
inv["days_of_stock"] = inv["stock"] / inv["daily_demand"].clip(lower=.01)
inv["status"] = np.select([inv["stock"] <= 0, inv["stock"] <= inv["reorder_level"], inv["days_of_stock"] < 14], ["Stockout", "Reorder", "At risk"], default="Healthy")
active = len(inv)
alerts = int(inv["status"].isin(["Stockout", "Reorder", "At risk"]).sum())
turnover = float(inv["units_sold"].sum() / inv["stock"].sum()) if inv["stock"].sum() else 0
risk = float((inv["status"] == "Stockout").mean() * 100)
cols = st.columns(4)
for col, label, value in [(cols[0], "Active SKUs", theme.format_number(active)), (cols[1], "Low stock alerts", theme.format_number(alerts)), (cols[2], "Turnover rate", f"{turnover:.2f}x"), (cols[3], "Stockout risk", theme.format_percent(risk))]:
    with col:
        st.metric(label, value)

health_tab, abc_tab, demand_tab = st.tabs(["Stock Health", "ABC Analysis", "Demand Forecast"])
with health_tab:
    left, right = st.columns([1, 2])
    with left:
        stockout = int((inv["stock"] <= 0).sum())
        fig = go.Figure(go.Indicator(mode="gauge+number", value=100 - risk, title={"text": "Stock health"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": theme.SUCCESS}, "bgcolor": theme.BG_ELEVATED, "bordercolor": theme.BORDER}))
        theme.show_chart(fig, 300)
    with right:
        depletion = inv.sort_values("days_of_stock").head(10)
        fig = px.bar(depletion, x="days_of_stock", y="name", orientation="h", color="status", title="Fastest depletion", labels={"days_of_stock": "Days of stock", "name": ""})
        theme.show_chart(fig, 300)
    st.markdown('<div class="nx-section">Urgent reorder queue</div>', unsafe_allow_html=True)
    st.dataframe(inv[inv["status"] != "Healthy"].sort_values(["status", "days_of_stock"])[["product_id", "name", "category", "stock", "reorder_level", "daily_demand", "days_of_stock", "status"]].head(50), use_container_width=True, hide_index=True)
with abc_tab:
    abc = inv.sort_values("revenue", ascending=False).copy()
    abc["cumulative_share"] = abc["revenue"].cumsum() / max(float(abc["revenue"].sum()), 1.0) * 100
    abc["class"] = np.select([abc["cumulative_share"] <= 80, abc["cumulative_share"] <= 95], ["A", "B"], default="C")
    grouped = abc.groupby(["category", "class"], as_index=False)["revenue"].sum()
    fig = px.bar(grouped, x="category", y="revenue", color="class", title="ABC revenue by category", labels={"revenue": "Revenue", "category": ""})
    theme.show_chart(fig, 350)
    pareto = abc.head(30)
    fig = go.Figure([go.Bar(x=pareto["name"], y=pareto["revenue"], name="Revenue", marker_color=theme.ACCENT), go.Scatter(x=pareto["name"], y=pareto["cumulative_share"], name="Cumulative %", yaxis="y2", line=dict(color=theme.WARNING))])
    fig.update_layout(title="Pareto curve", yaxis2={"overlaying": "y", "side": "right", "range": [0, 105]})
    theme.show_chart(fig, 370)
    st.dataframe(abc[["product_id", "name", "category", "revenue", "cumulative_share", "class"]].head(100), use_container_width=True, hide_index=True)
with demand_tab:
    selected = st.selectbox("Category", sorted(sales["category"].unique()))
    cat = sales[sales["category"] == selected].groupby("date", as_index=False)["quantity"].sum()
    if len(cat) >= 30:
        model = DemandForecaster(periods=30).fit(cat, date_col="date", value_col="quantity")
        pred = model.predict(30)
        fig = go.Figure([go.Scatter(x=cat["date"], y=cat["quantity"], name="Actual", line=dict(color=theme.ACCENT)), go.Scatter(x=pred["ds"], y=pred["yhat"], name="Forecast", line=dict(color=theme.SUCCESS, dash="dash")), go.Scatter(x=pred["ds"], y=pred["yhat_upper"], line=dict(width=0), showlegend=False), go.Scatter(x=pred["ds"], y=pred["yhat_lower"], fill="tonexty", fillcolor="rgba(34,197,94,.10)", line=dict(width=0), name="Confidence")])
        theme.show_chart(fig, 390)
        average = float(pred["yhat"].mean())
        st.metric("Recommended reorder horizon", f"{max(1, int(average * 14)):,} units", f"{selected} · 14 days")
    else:
        st.warning("At least 30 daily observations are required for a category forecast.")
    st.dataframe(cat.tail(30), use_container_width=True, hide_index=True)

theme.footer("NexusBI v2.1.0")
