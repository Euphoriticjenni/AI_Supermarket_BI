"""Lucciola premium demand forecasting page."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme

from src.forecasting import DemandForecaster, generate_forecast_insights

st.set_page_config(page_title="Lucciola", page_icon="◆", layout="wide")
theme.inject_theme()
theme.render_sidebar_nav("Forecasting")
theme.render_topbar(["Planning", "Demand Forecasting"])
st.title("Demand Forecasting")
st.caption("Prophet-based revenue and demand planning with transparent uncertainty bands.")


df = theme.load_data()
if df.empty:
    theme.render_empty_state("No history is available for forecasting.")
    st.stop()
# Validate required columns
_required = ['date', 'total_amount', 'category']
_missing = [c for c in _required if c not in df.columns]
if _missing:
    theme.render_empty_state(f"Data is missing required columns: {', '.join(_missing)}. Re-run the data pipeline.")
    st.stop()



@st.cache_data(show_spinner="Fitting demand model")
def fit_model(data: pd.DataFrame, horizon: int, category: str) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    subset = data if category == "All categories" else data[data["category"] == category]
    daily = subset.groupby("date", as_index=False)["total_amount"].sum()
    model = DemandForecaster(periods=horizon).fit(daily, date_col="date", value_col="total_amount")
    return model.predict(horizon), model.get_forecast_summary(), model.get_anomaly_dates(), model.get_components()


controls = st.columns([1, 1, 1, 1], gap="medium")
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


st.markdown(
    f'<div class="nx-panel"><span class="nx-badge nx-badge-success">Trained</span> &nbsp; {len(history):,} data points &nbsp; · &nbsp; Backend: {summary.get("backend", "Prophet")} &nbsp; · &nbsp; MAPE: {summary.get("mape", float("nan")):.1f}% &nbsp; · &nbsp; Last trained: {pd.Timestamp.now():%Y-%m-%d %H:%M}</div>',
    unsafe_allow_html=True,
)


# KPI — HTML cards (consistent, no st.metric alignment gap)
cols = st.columns(4, gap="medium")
for col, (label, value) in zip(cols, [
    ("Projected revenue", theme.format_currency(summary.get(f"next_{min(horizon, 30)}d", summary.get("next_30d")))),
    ("Next 7 days", theme.format_currency(summary.get("next_7d"))),
    ("Trend", str(summary.get("trend_direction", "stable")).title()),
    ("Confidence width", theme.format_currency(summary.get("confidence_width"))),
]):
    with col:
        st.markdown(f'<div class="nx-kpi"><div class="nx-kpi-label">{label}</div><div class="nx-kpi-value">{value}</div></div>', unsafe_allow_html=True)


# ── Main forecast — decluttered ──
fig = go.Figure()
# history faint area
fig.add_trace(go.Scatter(x=history["date"], y=history["Revenue"], line=dict(width=0), fill="tozeroy", fillcolor="rgba(59,130,246,0.05)", showlegend=False, hoverinfo="skip"))
# history line — spline
fig.add_trace(go.Scatter(x=history["date"], y=history["Revenue"], name="Historical", mode="lines", line=dict(color=theme.ACCENT, width=1.9, shape="spline", smoothing=0.85), hovertemplate="%{x|%b %d, %Y}<br>Actual: %{y:$,.0f}<extra></extra>"))
# confidence band — very light
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="90% band", fill="tonexty", fillcolor="rgba(34,197,94,.07)", line=dict(width=0), hovertemplate="Band: %{y:$,.0f}<extra></extra>"))
# forecast line
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", mode="lines", line=dict(color=theme.SUCCESS, dash="dash", width=2), hovertemplate="%{x|%b %d, %Y}<br>Forecast: %{y:$,.0f}<extra></extra>"))
# today divider — subtle
last_day = history["date"].max()
fig.add_vline(x=last_day, line_width=1, line_dash="dot", line_color="rgba(161,161,170,0.35)")
fig.add_annotation(x=last_day, y=1, yref="paper", text="today", showarrow=False, font=dict(color=theme.TEXT_MUTED, size=10), bgcolor="rgba(24,24,27,0.9)", bordercolor=theme.BORDER, borderpad=3, yshift=8)

fig.update_layout(title="Historical revenue and forward forecast", hovermode="x unified", legend=dict(orientation="h", y=1.04, x=0), margin=dict(t=42, b=24, l=12, r=12))
fig.update_xaxes(nticks=8, tickangle=0)
fig.update_yaxes(nticks=5, tickprefix="$", separatethousands=True)
theme.show_chart(fig, 480)


st.markdown('<div class="nx-section">Model components</div>', unsafe_allow_html=True)
left, right = st.columns(2, gap="large")
with left:
    trend = components["trend"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["ds"], y=trend["trend"], name="Trend", mode="lines", line=dict(color=theme.ACCENT, width=1.8, shape="spline", smoothing=0.8), fill="tozeroy", fillcolor="rgba(59,130,246,0.05)", hovertemplate="%{x|%b %d}<br>Trend: %{y:$,.0f}<extra></extra>"))
    fig.update_layout(title="Trend component", hovermode="x unified")
    fig.update_xaxes(nticks=6)
    fig.update_yaxes(nticks=4, tickprefix="$")
    theme.show_chart(fig, 300)
with right:
    # Weekly seasonality — aggregate to Mon–Sun bar, not a 90-day cluttered line
    weekly = components["weekly"].copy()
    weekly["dow"] = pd.to_datetime(weekly["ds"]).dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly["dow"] = pd.Categorical(weekly["dow"], categories=order, ordered=True)
    wk_bar = weekly.groupby("dow", as_index=False, observed=True)["weekly"].mean()
    fig = go.Figure(go.Bar(x=wk_bar["dow"], y=wk_bar["weekly"], marker_color=theme.COLORWAY[1], hovertemplate="%{x}: %{y:+.0f}<extra></extra>"))
    fig.update_layout(title="Weekly seasonality (avg lift)")
    fig.update_xaxes(tickangle=-18)
    fig.update_yaxes(nticks=4)
    theme.show_chart(fig, 300)


st.markdown('<div class="nx-section">Anomaly detection</div>', unsafe_allow_html=True)
if anomalies.empty:
    st.info("No historical dates are outside the model confidence band.")
else:
    anomaly_view = anomalies.copy()
    anomaly_view["deviation"] = (anomaly_view["actual"] - (anomaly_view["yhat_lower"] + anomaly_view["yhat_upper"]) / 2)
    anomaly_view["type"] = anomaly_view["anomaly_direction"].map({"above_expected": "Above expected", "below_expected": "Below expected"})
    st.dataframe(
        anomaly_view[["ds", "actual", "yhat_lower", "yhat_upper", "deviation", "type"]].sort_values("ds", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=240,
    )


insights = generate_forecast_insights({"summary": summary, "anomalies": anomalies})
with st.expander("Planning notes", expanded=True):
    for insight in insights[:5]:
        st.write(insight)


theme.footer("Forecasts use Prophet trend and seasonality with a 90% uncertainty interval. Estimates are planning aids, not commitments. · Lucciola v2.1.0")
