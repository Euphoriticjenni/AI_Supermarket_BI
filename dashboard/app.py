"""NexusBI Overview dashboard."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme

st.set_page_config(page_title="NexusBI", page_icon="◆", layout="wide")
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
    st.metric("Revenue", "—")
    st.plotly_chart(go.Figure(), use_container_width=True)
    st.plotly_chart(go.Figure(), use_container_width=True)
    st.dataframe(df, use_container_width=True)
    theme.footer("NexusBI v2.1.0")
    st.stop()

left, right = st.columns([3, 1])
with left:
    filtered = theme.date_filter(df)
with right:
    branch = st.selectbox("Branch", ["All"] + sorted(df["branch"].dropna().unique().tolist()))
if branch != "All":
    filtered = filtered[filtered["branch"] == branch]

revenue = float(filtered["total_amount"].sum())
transactions = int(filtered["transaction_id"].nunique())
avg_basket = revenue / transactions if transactions else 0.0
customers = int(filtered["customer_id"].nunique())
daily = filtered.groupby("date", as_index=False).agg(Revenue=("total_amount", "sum"), Transactions=("transaction_id", "nunique"))
trend = daily["Revenue"].tail(14).tolist()

cols = st.columns(4)
for col, label, value, delta, spark in [
    (cols[0], "Revenue", theme.format_currency(revenue), None, trend),
    (cols[1], "Transactions", theme.format_number(transactions), None, daily["Transactions"].tail(14).tolist()),
    (cols[2], "Avg basket", theme.format_currency(avg_basket), None, (daily["Revenue"] / daily["Transactions"].clip(lower=1)).tail(14).tolist()),
    (cols[3], "Customers", theme.format_number(customers), None, filtered.groupby("date")["customer_id"].nunique().tail(14).tolist()),
]:
    with col:
        st.markdown('<div class="nx-kpi">', unsafe_allow_html=True)
        st.metric(label, value, delta)
        st.markdown(theme.sparkline_svg(spark), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="nx-section">Revenue and mix</div>', unsafe_allow_html=True)
st.markdown('<div class="nx-section-sub">Daily performance and category contribution for the selected window.</div>', unsafe_allow_html=True)
left, right = st.columns([3, 2])
with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["Revenue"], mode="lines", name="Revenue", line=dict(color=theme.ACCENT, width=2), fill="tozeroy", fillcolor="rgba(59,130,246,.10)"))
    fig.update_layout(hovermode="x unified", title="Revenue trend")
    theme.show_chart(fig, 350)
with right:
    category = filtered.groupby("category", as_index=False)["total_amount"].sum().sort_values("total_amount")
    fig = px.bar(category, x="total_amount", y="category", orientation="h", title="Revenue by category", labels={"total_amount": "Revenue", "category": ""})
    theme.show_chart(fig, 350)

left, right = st.columns(2)
with left:
    hourly = filtered.groupby("hour_of_day", as_index=False)["total_amount"].sum()
    fig = px.bar(hourly, x="hour_of_day", y="total_amount", title="Hourly traffic", labels={"hour_of_day": "Hour", "total_amount": "Revenue"})
    theme.show_chart(fig, 300)
with right:
    payment = filtered["payment_method"].value_counts().reset_index()
    payment.columns = ["payment_method", "transactions"]
    fig = px.pie(payment, names="payment_method", values="transactions", hole=.55, title="Payment mix")
    theme.show_chart(fig, 300)

st.markdown('<div class="nx-section">Recent transactions</div>', unsafe_allow_html=True)
recent = filtered.sort_values("datetime", ascending=False).head(20).copy()
recent["total_amount"] = recent["total_amount"].map(theme.format_currency)
recent["datetime"] = recent["datetime"].dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(recent[["datetime", "transaction_id", "category", "branch", "payment_method", "quantity", "total_amount"]], use_container_width=True, hide_index=True)
theme.footer("NexusBI v2.1.0")
