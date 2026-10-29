"""Lucciola Sales Analytics."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme

from src.forecasting import DemandForecaster

st.set_page_config(page_title="Lucciola", page_icon="◆", layout="wide")
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
    st.stop()
# Validate required columns
_required = ['transaction_id', 'date', 'branch', 'category', 'payment_method', 'total_amount', 'profit', 'quantity', 'product_id', 'product_name']
_missing = [c for c in _required if c not in df.columns]
if _missing:
    theme.render_empty_state(f"Data is missing required columns: {', '.join(_missing)}. Re-run the data pipeline.")
    st.stop()


c1, c2, c3, c4 = st.columns(4, gap="medium")
with c1:
    branch = st.selectbox("Branch", ["All"] + sorted(df["branch"].unique()))
with c2:
    category = st.selectbox("Category", ["All"] + sorted(df["category"].unique()))
with c3:
    payment = st.selectbox("Payment", ["All"] + sorted(df["payment_method"].unique()))
with c4:
    filtered = theme.date_filter(df)
if branch != "All":
    filtered = filtered[filtered["branch"] == branch]
if category != "All":
    filtered = filtered[filtered["category"] == category]
if payment != "All":
    filtered = filtered[filtered["payment_method"] == payment]
if filtered.empty:
    theme.render_empty_state("No data matches the selected filters. Try broadening the date range or filter selections.")
    st.stop()

revenue = float(filtered["total_amount"].sum())
orders = int(filtered["transaction_id"].nunique())
aov = revenue / orders if orders else 0
profit = float(filtered["profit"].sum())

# KPI — HTML cards for alignment consistency with Overview
kcols = st.columns(4, gap="medium")
for col, (label, value) in zip(kcols, [
    ("Revenue", theme.format_currency(revenue)),
    ("Transactions", theme.format_number(orders)),
    ("Avg basket", theme.format_currency(aov)),
    ("Profit", theme.format_currency(profit)),
]):
    with col:
        st.markdown(f'<div class="nx-kpi"><div class="nx-kpi-label">{label}</div><div class="nx-kpi-value">{value}</div></div>', unsafe_allow_html=True)

trend_tab, breakdown_tab, distribution_tab = st.tabs(["Trend", "Breakdown", "Distribution"])
with trend_tab:
    daily = filtered.groupby("date", as_index=False)["total_amount"].sum().rename(columns={"total_amount": "Revenue"})
    daily["MA 7d"] = daily["Revenue"].rolling(7, min_periods=1, center=True).mean()
    daily["MA 30d"] = daily["Revenue"].rolling(30, min_periods=1, center=True).mean()
    # declutter: weekly resample when window is large
    use_weekly = len(daily) > 90
    plot_daily = daily.copy()
    if use_weekly:
        plot_daily["wk"] = plot_daily["date"].dt.to_period("W").dt.start_time
        plot_daily = plot_daily.groupby("wk", as_index=False).agg(Revenue=("Revenue", "sum"), **{"MA 7d": ("MA 7d", "mean"), "MA 30d": ("MA 30d", "mean")}).rename(columns={"wk": "date"})

    forecast_on = st.checkbox("Show Prophet forecast", value=False)

    fig = go.Figure()
    # faint area under revenue
    fig.add_trace(go.Scatter(x=plot_daily["date"], y=plot_daily["Revenue"], line=dict(width=0), fill="tozeroy", fillcolor="rgba(59,130,246,0.06)", showlegend=False, hoverinfo="skip"))
    # revenue line
    fig.add_trace(go.Scatter(x=plot_daily["date"], y=plot_daily["Revenue"], name="Revenue", mode="lines", line=dict(color=theme.ACCENT, width=2, shape="spline", smoothing=0.85), hovertemplate="%{x|%b %d, %Y}<br>%{y:$,.0f}<extra></extra>"))
    # MAs — muted, dashed
    fig.add_trace(go.Scatter(x=plot_daily["date"], y=plot_daily["MA 7d"], name="7d avg", mode="lines", line=dict(color="#A1A1AA", width=1.3, dash="dot"), hovertemplate="7d: %{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=plot_daily["date"], y=plot_daily["MA 30d"], name="30d avg", mode="lines", line=dict(color="#71717A", width=1.2, dash="dash"), hovertemplate="30d: %{y:$,.0f}<extra></extra>"))
    if forecast_on and len(daily) >= 30:
        model = DemandForecaster(periods=30).fit(daily, date_col="date", value_col="Revenue")
        pred = model.predict(30)
        fig.add_trace(go.Scatter(x=pred["ds"], y=pred["yhat_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pred["ds"], y=pred["yhat_lower"], fill="tonexty", fillcolor="rgba(34,197,94,.08)", line=dict(width=0), name="Confidence", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pred["ds"], y=pred["yhat"], name="Prophet forecast", line=dict(color=theme.SUCCESS, dash="dash", width=2), hovertemplate="%{x|%b %d}<br>Proj: %{y:$,.0f}<extra></extra>"))
    fig.update_layout(title="Revenue trend" + (" · weekly" if use_weekly else ""), hovermode="x unified", legend=dict(orientation="h", y=1.04, x=0), margin=dict(t=42, b=24, l=12, r=12))
    fig.update_xaxes(nticks=8, tickangle=0)
    fig.update_yaxes(nticks=5, separatethousands=True, tickprefix="$")
    theme.show_chart(fig, 420)
    st.dataframe(daily.tail(30), use_container_width=True, hide_index=True, height=220)

with breakdown_tab:
    by_time = filtered.assign(month=filtered["date"].dt.to_period("M").astype(str)).groupby(["month", "category"], as_index=False)["total_amount"].sum()
    fig = px.bar(by_time, x="month", y="total_amount", color="category", title="Category revenue over time", labels={"total_amount": "Revenue", "month": "Month"})
    fig.update_layout(barmode="stack", barnorm=None, legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(tickangle=-22, nticks=10)
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    theme.show_chart(fig, 380)
    by_product = filtered.groupby(["product_id", "product_name", "category"], as_index=False)["total_amount"].sum().sort_values("total_amount", ascending=False).head(25)
    fig = px.treemap(by_product, path=["category", "product_name"], values="total_amount", title="Top 25 products by revenue")
    fig.update_traces(hovertemplate="%{label}<br>$%{value:,.0f}<extra></extra>", textinfo="label+value")
    theme.show_chart(fig, 400)

with distribution_tab:
    left, right = st.columns(2, gap="large")
    with left:
        fig = px.histogram(filtered, x="total_amount", nbins=40, title="Transaction amount", labels={"total_amount": "Amount"}, color_discrete_sequence=[theme.ACCENT])
        fig.update_layout(margin=dict(t=38, b=24))
        fig.update_xaxes(nticks=8, tickprefix="$")
        fig.update_yaxes(nticks=5)
        theme.show_chart(fig, 340)
    with right:
        fig = px.box(filtered, x="category", y="total_amount", color="category", title="Basket by category", labels={"total_amount": "Amount", "category": ""}, points="outliers")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(tickprefix="$", nticks=5)
        fig.update_xaxes(tickangle=-12)
        theme.show_chart(fig, 340)
    st.dataframe(filtered.sort_values("datetime", ascending=False).head(100), use_container_width=True, hide_index=True, height=360)

st.markdown('<div class="nx-section">Transaction detail</div>', unsafe_allow_html=True)
st.dataframe(filtered.sort_values("datetime", ascending=False).head(50), use_container_width=True, hide_index=True, height=260)
theme.footer("Lucciola v2.1.0")
