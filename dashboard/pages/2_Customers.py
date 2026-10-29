"""Lucciola Customer Analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme

from src.forecasting import DemandForecaster

st.set_page_config(page_title="Lucciola", page_icon="◆", layout="wide")
theme.inject_theme()
theme.render_sidebar_nav("Customers")
theme.render_topbar(["Analytics", "Customers"])
st.title("Customer Analytics")
st.caption("Understand customer value, retention, and the segments behind revenue.")


@st.cache_data(show_spinner="Loading customer analytics")
def load_customer_data():
    return theme.load_data()


df = load_customer_data()
if df.empty:
    theme.render_empty_state("No customer activity is available.")
    st.metric("Customers", "—")
    st.plotly_chart(go.Figure(), use_container_width=True)
    st.plotly_chart(go.Figure(), use_container_width=True)
    st.dataframe(df, use_container_width=True)
    st.stop()
# Validate required columns
_required = ['customer_id', 'date', 'transaction_id', 'total_amount']
_missing = [c for c in _required if c not in df.columns]
if _missing:
    theme.render_empty_state(f"Data is missing required columns: {', '.join(_missing)}. Re-run the data pipeline.")
    st.stop()


snapshot = df["date"].max()
rfm = df.groupby("customer_id").agg(
    LastPurchase=("date", "max"),
    Frequency=("transaction_id", "nunique"),
    Monetary=("total_amount", "sum"),
).reset_index()
rfm["Recency"] = (snapshot - rfm["LastPurchase"]).dt.days
for source, target in [("Recency", "R"), ("Frequency", "F"), ("Monetary", "M")]:
    ranks = rfm[source].rank(method="first", pct=True)
    rfm[target] = np.ceil(ranks * 5).clip(1, 5).astype(int)
rfm["segment"] = np.select(
    [
        (rfm.R >= 4) & (rfm.F >= 4) & (rfm.M >= 4),
        (rfm.R >= 3) & (rfm.F >= 3),
        (rfm.R <= 2) & (rfm.F >= 3),
        (rfm.R <= 2) & (rfm.F <= 2),
    ],
    ["Champions", "Loyal", "At-Risk", "Lost"],
    default="Potential",
)
rfm["ltv"] = (rfm["Monetary"] / rfm["Frequency"].clip(lower=1) * rfm["Frequency"] * 1.5).round(2)

new_month = snapshot.to_period("M")
new_customers = int((rfm["LastPurchase"].dt.to_period("M") == new_month).sum())
repeat_rate = float((rfm["Frequency"] > 1).mean() * 100)
avg_ltv = float(rfm["ltv"].mean())
cols = st.columns(4)
for col, label, value in [
    (cols[0], "Total customers", theme.format_number(len(rfm))),
    (cols[1], "New this month", theme.format_number(new_customers)),
    (cols[2], "Repeat rate", theme.format_percent(repeat_rate)),
    (cols[3], "Avg LTV", theme.format_currency(avg_ltv)),
]:
    with col:
        st.metric(label, value)

segment_tab, cohort_tab, top_tab = st.tabs(["Segments", "Cohorts", "Top Customers"])
with segment_tab:
    counts = rfm["segment"].value_counts().rename_axis("segment").reset_index(name="customers")
    left, right = st.columns(2)
    with left:
        fig = px.pie(counts, names="segment", values="customers", hole=0.55, title="Customer segment mix")
        theme.show_chart(fig, 340)
    with right:
        fig = px.scatter(rfm, x="Frequency", y="Monetary", color="segment", size="ltv", hover_name="customer_id", title="Frequency and monetary value", labels={"Monetary": "Spend"})
        theme.show_chart(fig, 340)
    st.dataframe(counts.assign(share=counts["customers"] / len(rfm) * 100), use_container_width=True, hide_index=True)
with cohort_tab:
    base = df[["customer_id", "date"]].copy()
    first = base.groupby("customer_id")["date"].min().rename("cohort")
    base = base.join(first, on="customer_id")
    base["cohort_month"] = base["cohort"].dt.to_period("M").astype(str)
    base["activity_month"] = base["date"].dt.to_period("M").astype(str)
    cohort = base.groupby(["cohort_month", "activity_month"])["customer_id"].nunique().reset_index()
    cohort["retention"] = cohort.groupby("cohort_month")["customer_id"].transform(lambda s: s / s.iloc[0] * 100)
    heat = cohort.pivot(index="cohort_month", columns="activity_month", values="retention").fillna(0)
    fig = px.imshow(heat, color_continuous_scale=[[0, "#18181B"], [1, theme.ACCENT]], aspect="auto", title="Monthly customer retention", labels={"color": "Retention %"})
    theme.show_chart(fig, 420)
    st.dataframe(heat.round(1), use_container_width=True)
    daily_customers = df.groupby("date")["customer_id"].nunique().reset_index(name="customers")
    fig = px.line(daily_customers, x="date", y="customers", title="Customer growth", labels={"customers": "Active customers"})
    theme.show_chart(fig, 300)
with top_tab:
    top = rfm.sort_values("ltv", ascending=False).head(25).copy()
    top["Monetary"] = top["Monetary"].map(theme.format_currency)
    top["ltv"] = top["ltv"].map(theme.format_currency)
    st.dataframe(top[["customer_id", "segment", "Frequency", "Monetary", "ltv", "Recency"]], use_container_width=True, hide_index=True)
    daily = df.groupby("date")["customer_id"].nunique().reset_index(name="customers")
    if len(daily) >= 30:
        model = DemandForecaster(periods=30).fit(daily, date_col="date", value_col="customers")
        pred = model.predict(30)
        fig = go.Figure([
            go.Scatter(x=daily["date"], y=daily["customers"], name="Actual", line=dict(color=theme.ACCENT)),
            go.Scatter(x=pred["ds"], y=pred["yhat"], name="Forecast", line=dict(color=theme.SUCCESS, dash="dash")),
        ])
    else:
        fig = px.line(daily, x="date", y="customers", title="Customer growth")
    theme.show_chart(fig, 340)
    st.dataframe(df.sort_values("datetime", ascending=False).head(30), use_container_width=True, hide_index=True)

theme.footer("Lucciola v2.1.0")