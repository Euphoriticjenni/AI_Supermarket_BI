"""NexusBI Sales Analytics."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme
from src.forecasting import DemandForecaster

st.set_page_config(page_title="NexusBI · Sales", page_icon="◆", layout="wide")
theme.inject_theme()
theme.render_sidebar_nav("Sales")
theme.render_topbar(["Analytics", "Sales"])
st.title("Sales Analytics")
st.caption("Explore commercial performance by time, branch, category, and payment behavior.")

@st.cache_data(show_spinner="Loading sales analytics")
def load_sales_data():
    return theme.load_data()

df = load_sales_data()
if df.empty:
    theme.render_empty_state("No sales data is available.")
    st.metric("Revenue", "—")
    st.plotly_chart(go.Figure(), use_container_width=True)
    st.plotly_chart(go.Figure(), use_container_width=True)
    st.dataframe(df, use_container_width=True)
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    branch = st.selectbox("Branch", ["All"] + sorted(df["branch"].unique()))
with c2:
    category = st.selectbox("Category", ["All"] + sorted(df["category"].unique()))
with c3:
    payment = st.selectbox("Payment", ["All"] + sorted(df["payment_method"].unique()))
with c4:
    filtered = theme.date_filter(df)
if branch != "All": filtered = filtered[filtered["branch"] == branch]
if category != "All": filtered = filtered[filtered["category"] == category]
if payment != "All": filtered = filtered[filtered["payment_method"] == payment]

revenue = float(filtered["total_amount"].sum())
orders = int(filtered["transaction_id"].nunique())
aov = revenue / orders if orders else 0
profit = float(filtered["profit"].sum())
cols = st.columns(4)
for col, label, value in [(cols[0], "Revenue", theme.format_currency(revenue)), (cols[1], "Transactions", theme.format_number(orders)), (cols[2], "Avg basket", theme.format_currency(aov)), (cols[3], "Profit", theme.format_currency(profit))]:
    with col:
        st.metric(label, value)

trend_tab, breakdown_tab, distribution_tab = st.tabs(["Trend", "Breakdown", "Distribution"])
with trend_tab:
    daily = filtered.groupby("date", as_index=False)["total_amount"].sum().rename(columns={"total_amount": "Revenue"})
    daily["MA 7d"] = daily["Revenue"].rolling(7, min_periods=1).mean()
    daily["MA 30d"] = daily["Revenue"].rolling(30, min_periods=1).mean()
    forecast_on = st.checkbox("Show Prophet forecast", value=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["Revenue"], name="Revenue", line=dict(color=theme.ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["MA 7d"], name="7d average", line=dict(color=theme.COLORWAY[1], width=1.5)))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["MA 30d"], name="30d average", line=dict(color=theme.COLORWAY[2], width=1.5)))
    if forecast_on and len(daily) >= 30:
        model = DemandForecaster(periods=30).fit(daily, date_col="date", value_col="Revenue")
        pred = model.predict(30)
        fig.add_trace(go.Scatter(x=pred["ds"], y=pred["yhat_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pred["ds"], y=pred["yhat_lower"], fill="tonexty", fillcolor="rgba(34,197,94,.10)", line=dict(width=0), name="Confidence", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pred["ds"], y=pred["yhat"], name="Prophet forecast", line=dict(color=theme.SUCCESS, dash="dash", width=2)))
    theme.show_chart(fig, 430)
    st.dataframe(daily.tail(30), use_container_width=True, hide_index=True)
with breakdown_tab:
    by_time = filtered.assign(month=filtered["date"].dt.to_period("M").astype(str)).groupby(["month", "category"], as_index=False)["total_amount"].sum()
    fig = px.bar(by_time, x="month", y="total_amount", color="category", title="Category revenue over time", labels={"total_amount": "Revenue", "month": "Month"})
    theme.show_chart(fig, 360)
    by_product = filtered.groupby(["product_id", "product_name", "category"], as_index=False)["total_amount"].sum().sort_values("total_amount", ascending=False).head(40)
    fig = px.treemap(by_product, path=["category", "product_name"], values="total_amount", title="Product revenue treemap")
    theme.show_chart(fig, 380)
with distribution_tab:
    left, right = st.columns(2)
    with left:
        fig = px.histogram(filtered, x="total_amount", nbins=50, title="Transaction amount distribution", labels={"total_amount": "Amount"})
        theme.show_chart(fig, 340)
    with right:
        fig = px.box(filtered, x="category", y="total_amount", color="category", title="Basket distribution by category", labels={"total_amount": "Amount", "category": ""})
        theme.show_chart(fig, 340)
    st.dataframe(filtered.sort_values("datetime", ascending=False).head(100), use_container_width=True, hide_index=True)

st.markdown('<div class="nx-section">Transaction detail</div>', unsafe_allow_html=True)
st.dataframe(filtered.sort_values("datetime", ascending=False).head(50), use_container_width=True, hide_index=True)
theme.footer("NexusBI v2.1.0")
