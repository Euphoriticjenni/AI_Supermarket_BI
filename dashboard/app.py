"""Lucciola Overview dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme

st.set_page_config(page_title="Lucciola", page_icon="◆", layout="wide")
theme.inject_theme()
theme.render_sidebar_nav("Overview")
theme.render_topbar(["Overview"])
st.title("Overview")
st.caption("A current view of revenue, traffic, customers, and basket performance.")


@st.cache_data(show_spinner="Loading overview")
def load_overview_data():
    return theme.load_data()


df = load_overview_data()
if df.empty:
    theme.render_empty_state("No sales data is available. Run the data pipeline first.")
    st.stop()
# Validate required columns
_required = ['transaction_id', 'date', 'branch', 'category', 'payment_method', 'quantity', 'total_amount', 'customer_id', 'datetime', 'hour_of_day']
_missing = [c for c in _required if c not in df.columns]
if _missing:
    theme.render_empty_state(f"Data is missing required columns: {', '.join(_missing)}. Re-run the data pipeline.")
    st.stop()

# ── Filters row: date + branch aligned ──
f1, f2 = st.columns([3, 1], gap="large")
with f1:
    filtered = theme.date_filter(df)
with f2:
    branch = st.selectbox("Branch", ["All"] + sorted(df["branch"].dropna().unique().tolist()))
if branch != "All":
    filtered = filtered[filtered["branch"] == branch]
if filtered.empty:
    theme.render_empty_state("No data matches the selected filters. Try broadening the date range or branch filter.")
    st.stop()

# ── KPIs ──
revenue = float(filtered["total_amount"].sum())
transactions = int(filtered["transaction_id"].nunique())
avg_basket = revenue / transactions if transactions else 0.0
customers = int(filtered["customer_id"].nunique())
daily = filtered.groupby("date", as_index=False).agg(Revenue=("total_amount", "sum"), Transactions=("transaction_id", "nunique"))
trend = daily["Revenue"].tail(14).tolist()

# Use HTML cards for pixel-perfect alignment (no st.metric nesting issue)
kcols = st.columns(4, gap="medium")
kpi_items = [
    ("Revenue", theme.format_currency(revenue), trend, theme.ACCENT),
    ("Transactions", theme.format_number(transactions), daily["Transactions"].tail(14).tolist(), theme.COLORWAY[2]),
    ("Avg basket", theme.format_currency(avg_basket), (daily["Revenue"] / daily["Transactions"].clip(lower=1)).tail(14).tolist(), theme.COLORWAY[1]),
    ("Customers", theme.format_number(customers), filtered.groupby("date")["customer_id"].nunique().tail(14).tolist(), theme.COLORWAY[3]),
]
for col, (label, value, spark, color) in zip(kcols, kpi_items):
    with col:
        spark_svg = theme.sparkline_svg(spark, color=color) if spark else ""
        st.markdown(
            f'<div class="nx-kpi"><div><div class="nx-kpi-label">{label}</div><div class="nx-kpi-value">{value}</div></div>{spark_svg}</div>',
            unsafe_allow_html=True,
        )

# ── Revenue and mix ──
st.markdown('<div class="nx-section">Revenue and mix</div>', unsafe_allow_html=True)
st.markdown('<div class="nx-section-sub">Daily performance and category contribution for the selected window.</div>', unsafe_allow_html=True)
left, right = st.columns([3, 2], gap="large")
with left:
    # Declutter: resample to weekly when window is very wide, smooth daily with 7d MA
    daily_sorted = daily.sort_values("date").copy()
    n_days = len(daily_sorted)
    # choose plot granularity
    if n_days > 90:
        # weekly aggregation to reduce points ~13x
        wk = filtered.copy()
        wk["wk"] = wk["date"].dt.to_period("W").dt.start_time
        plot_df = wk.groupby("wk", as_index=False).agg(Revenue=("total_amount", "sum")).rename(columns={"wk": "date"})
        mode = "weekly"
    else:
        plot_df = daily_sorted
        mode = "daily"

    fig = go.Figure()
    # main line — thinner, no fill to reduce clutter
    fig.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df["Revenue"],
        mode="lines", name="Revenue",
        line=dict(color=theme.ACCENT, width=2, shape="spline", smoothing=0.9),
        hovertemplate="%{x|%b %d, %Y}<br>Revenue: %{y:$,.0f}<extra></extra>",
    ))
    # 7-day moving average overlay when daily and enough points
    if mode == "daily" and n_days >= 14:
        ma7 = daily_sorted["Revenue"].rolling(7, min_periods=1, center=True).mean()
        fig.add_trace(go.Scatter(
            x=daily_sorted["date"], y=ma7,
            mode="lines", name="7-day avg",
            line=dict(color="rgba(161,161,170,0.95)", width=1.4, dash="dot"),
            hovertemplate="7d avg: %{y:$,.0f}<extra></extra>",
        ))
    # subtle area under main line
    fig.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df["Revenue"],
        mode="none", fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        title=dict(text="Revenue trend" + (" · weekly" if mode == "weekly" else ""), font=dict(size=13, color=theme.TEXT_PRIMARY)),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=0),
        margin=dict(l=12, r=12, t=42, b=24),
    )
    fig.update_xaxes(tickformat="%b %d", nticks=7, tickangle=0, showgrid=True, gridcolor="rgba(39,39,42,0.45)")
    fig.update_yaxes(tickprefix="$", separatethousands=True, showgrid=True, gridcolor="rgba(39,39,42,0.45)", nticks=5)
    theme.show_chart(fig, 360)
with right:
    category = filtered.groupby("category", as_index=False)["total_amount"].sum().sort_values("total_amount")
    fig = px.bar(category, x="total_amount", y="category", orientation="h", title="Revenue by category",
                 labels={"total_amount": "Revenue", "category": ""}, text_auto=".2s")
    fig.update_traces(marker_line_width=0, hovertemplate="%{y}: %{x:$,.0f}<extra></extra>")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    theme.show_chart(fig, 360)


left, right = st.columns(2, gap="large")
with left:
    hourly = filtered.groupby("hour_of_day", as_index=False)["total_amount"].sum()
    fig = px.bar(hourly, x="hour_of_day", y="total_amount", title="Hourly traffic",
                 labels={"hour_of_day": "Hour", "total_amount": "Revenue"})
    fig.update_traces(hovertemplate="Hour %{x}: %{y:$,.0f}<extra></extra>")
    fig.update_xaxes(dtick=2, nticks=12)
    theme.show_chart(fig, 300)
with right:
    payment = filtered["payment_method"].value_counts().reset_index()
    payment.columns = ["payment_method", "transactions"]
    fig = px.pie(payment, names="payment_method", values="transactions", hole=0.58, title="Payment mix")
    fig.update_traces(textinfo="percent+label", textposition="outside", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>")
    theme.show_chart(fig, 300)


st.markdown('<div class="nx-section">Recent transactions</div>', unsafe_allow_html=True)
recent = filtered.sort_values("datetime", ascending=False).head(20).copy()
recent["total_amount"] = recent["total_amount"].map(theme.format_currency)
recent["datetime"] = recent["datetime"].dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(recent[["datetime", "transaction_id", "category", "branch", "payment_method", "quantity", "total_amount"]],
             use_container_width=True, hide_index=True, height=380)
theme.footer("Lucciola v2.1.0")
