"""NexusBI premium demand forecasting page."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme
from src.forecasting import DemandForecaster, generate_forecast_insights

st.set_page_config(page_title="NexusBI · Forecasting", page_icon="◆", layout="wide")
theme.inject_theme()
theme.render_sidebar_nav("Forecasting")
theme.render_topbar(["Planning", "Demand Forecasting"])
st.title("Demand Forecasting")
st.caption("Prophet-based revenue and demand planning with transparent uncertainty bands.")

df = theme.load_data()
if df.empty:
    theme.render_empty_state("No history is available for forecasting.")
    st.metric("Projected revenue", "—")
    st.plotly_chart(go.Figure(), use_container_width=True)
    st.plotly_chart(go.Figure(), use_container_width=True)
    st.dataframe(df, use_container_width=True)
    st.stop()

@st.cache_data(show_spinner="Fitting demand model")
def fit_model(data: pd.DataFrame, horizon: int, category: str) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    subset = data if category == "All categories" else data[data["category"] == category]
    daily = subset.groupby("date", as_index=False)["total_amount"].sum()
    model = DemandForecaster(periods=horizon).fit(daily, date_col="date", value_col="total_amount")
    return model.predict(horizon), model.get_forecast_summary(), model.get_anomaly_dates(), model.get_components()

controls = st.columns([1, 1, 1, 1])
with controls[0]:
    horizon = st.slider("Horizon", 7, 180, 30)
with controls[1]:
    granularity = st.radio("Granularity", ["Daily", "Weekly"], horizontal=True)
with controls[2]:
    category = st.selectbox("Category", ["All categories"] + sorted(df["category"].unique()))
with controls[3]:
    st.write("")
    retrain = st.button("Retrain model", type="primary", use_container_width=True)
if retrain:
    fit_model.clear()

forecast, summary, anomalies, components = fit_model(df, horizon, category)
subset = df if category == "All categories" else df[df["category"] == category]
history = subset.groupby("date", as_index=False)["total_amount"].sum().rename(columns={"total_amount": "Revenue"})
if granularity == "Weekly":
    history = history.set_index("date").resample("W")["Revenue"].sum().reset_index()

st.markdown(f'<div class="nx-panel"><span class="nx-badge nx-badge-success">Trained</span> &nbsp; {len(history):,} data points &nbsp; · &nbsp; Backend: {summary.get("backend", "Prophet")} &nbsp; · &nbsp; MAPE: {summary.get("mape", float("nan")):.1f}% &nbsp; · &nbsp; Last trained: {pd.Timestamp.now():%Y-%m-%d %H:%M}</div>', unsafe_allow_html=True)

cols = st.columns(4)
for col, label, value in [(cols[0], "Projected revenue", theme.format_currency(summary.get(f"next_{min(horizon, 30)}d", summary.get("next_30d")))), (cols[1], "Next 7 days", theme.format_currency(summary.get("next_7d"))), (cols[2], "Trend", str(summary.get("trend_direction", "stable")).title()), (cols[3], "Confidence width", theme.format_currency(summary.get("confidence_width")))]:
    with col:
        st.metric(label, value)

fig = go.Figure()
fig.add_trace(go.Scatter(x=history["date"], y=history["Revenue"], name="Historical", line=dict(color=theme.ACCENT, width=2)))
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Confidence", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="Confidence band", fill="tonexty", fillcolor="rgba(34,197,94,.10)", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color=theme.SUCCESS, dash="dash", width=2)))
last_day = history["date"].max()
fig.add_shape(type="line", x0=last_day, x1=last_day, y0=0, y1=1, yref="paper", line=dict(dash="dash", color=theme.TEXT_MUTED))
fig.add_annotation(x=last_day, y=1, yref="paper", text="today", showarrow=False, font=dict(color=theme.TEXT_MUTED))
fig.update_layout(title="Historical revenue and forward forecast", hovermode="x unified")
theme.show_chart(fig, 500)

st.markdown('<div class="nx-section">Model components</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    trend = components["trend"]
    fig = go.Figure(go.Scatter(x=trend["ds"], y=trend["trend"], name="Trend", line=dict(color=theme.ACCENT)))
    fig.update_layout(title="Trend component")
    theme.show_chart(fig, 300)
with right:
    weekly = components["weekly"]
    fig = go.Figure(go.Scatter(x=weekly["ds"], y=weekly["weekly"], name="Weekly", line=dict(color=theme.COLORWAY[1])))
    fig.update_layout(title="Weekly seasonality")
    theme.show_chart(fig, 300)

st.markdown('<div class="nx-section">Anomaly detection</div>', unsafe_allow_html=True)
if anomalies.empty:
    st.info("No historical dates are outside the model confidence band.")
else:
    anomaly_view = anomalies.copy()
    anomaly_view["deviation"] = (anomaly_view["actual"] - (anomaly_view["yhat_lower"] + anomaly_view["yhat_upper"]) / 2)
    anomaly_view["type"] = anomaly_view["anomaly_direction"].map({"above_expected": "Above expected", "below_expected": "Below expected"})
    st.dataframe(anomaly_view[["ds", "actual", "yhat_lower", "yhat_upper", "deviation", "type"]].sort_values("ds", ascending=False), use_container_width=True, hide_index=True)

insights = generate_forecast_insights({"summary": summary, "anomalies": anomalies})
with st.expander("Planning notes", expanded=True):
    for insight in insights[:5]:
        st.write(insight)

theme.footer("Forecasts use Prophet trend and seasonality with a 90% uncertainty interval. Estimates are planning aids, not commitments. · NexusBI v2.1.0")
