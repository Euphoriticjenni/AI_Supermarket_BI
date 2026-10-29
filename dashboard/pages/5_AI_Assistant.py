"""Lucciola AI Assistant workspace."""
from __future__ import annotations

import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import theme

from src.ai_assistant import BusinessAssistant
from src.forecasting import DemandForecaster
from src.report_generator import generate_executive_report

st.set_page_config(page_title="Lucciola", page_icon="◆", layout="wide")
theme.inject_theme()
theme.render_sidebar_nav("AI Assistant")
theme.render_topbar(["Workspace", "AI Assistant"])
st.title("AI Assistant")
st.caption("Ask operational questions in plain language and get answers grounded in Lucciola data.")


@st.cache_data(show_spinner="Loading assistant context")
def load_assistant_data() -> pd.DataFrame:
    return theme.load_data()


df = load_assistant_data()
try:
    assistant = BusinessAssistant()
except Exception as exc:
    assistant = None
    st.warning(f"The query assistant is unavailable: {exc}")


if "nexus_chat" not in st.session_state:
    st.session_state.nexus_chat = []
if "nexus_last_result" not in st.session_state:
    st.session_state.nexus_last_result = None


suggestions = [
    "What was total revenue by branch?",
    "Which products need restocking?",
    "Show the monthly revenue trend",
    "What are the peak shopping hours?",
]


def answer_question(question: str) -> None:
    question = question.strip()
    if not question:
        return
    answer = "I could not connect to the business query engine. Try a revenue, product, stock, or payment question."
    data = pd.DataFrame()
    if assistant is not None:
        try:
            result = assistant.respond(question) if hasattr(assistant, "respond") else assistant.ask(question)
            answer = result.get("answer", str(result)) if isinstance(result, dict) else str(result)
            data = result.get("data", pd.DataFrame()) if isinstance(result, dict) else pd.DataFrame()
        except Exception as exc:
            answer = f"The assistant could not complete that query: {exc}"
    if "forecast" in question.lower() or "projection" in question.lower():
        daily = df.groupby("date", as_index=False)["total_amount"].sum()
        if len(daily) >= 30:
            model = DemandForecaster(periods=30).fit(daily, date_col="date", value_col="total_amount")
            summary = model.get_forecast_summary()
            answer += f" Forecast signal: the next 30 days are projected at {theme.format_currency(summary.get('next_30d'))}, with an {summary.get('trend_direction')} trend."
    st.session_state.nexus_chat.append({"role": "user", "text": question})
    st.session_state.nexus_chat.append({"role": "assistant", "text": answer})
    st.session_state.nexus_last_result = data


left, right = st.columns([7, 3], gap="large")
with left:
    st.markdown('<div class="nx-section">Conversation</div>', unsafe_allow_html=True)
    for message in st.session_state.nexus_chat[-12:]:
        color = theme.ACCENT if message["role"] == "user" else theme.BG_SURFACE
        align = "right" if message["role"] == "user" else "left"
        st.markdown(
            f'<div style="background:{color};color:{theme.TEXT_PRIMARY};border:1px solid {theme.BORDER};border-radius:8px;padding:12px 14px;margin:8px 0;text-align:{align}"><strong>{"You" if message["role"] == "user" else "Lucciola"}</strong><br>{message["text"]}</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div style="margin-top:12px;color:#A1A1AA;font-size:.78rem">Suggested questions</div>', unsafe_allow_html=True)
    chips = st.columns(4, gap="small")
    for col, suggestion in zip(chips, suggestions):
        with col:
            if st.button(suggestion, key=f"suggestion_{suggestion}", use_container_width=True):
                answer_question(suggestion)
    question = st.text_input("Ask a question", placeholder="Revenue, stock, customers, or forecasts")
    if st.button("Send question", type="primary") and question:
        answer_question(question)
    if st.session_state.nexus_chat:
        if st.button("Clear conversation"):
            st.session_state.nexus_chat = []
            st.rerun()


with right:
    st.markdown('<div class="nx-section">Insights</div>', unsafe_allow_html=True)
    revenue = float(df["total_amount"].sum()) if not df.empty else 0
    k1, k2 = st.columns(2, gap="small")
    with k1:
        st.markdown(f'<div class="nx-kpi" style="min-height:96px"><div class="nx-kpi-label">Revenue in context</div><div class="nx-kpi-value" style="font-size:1.25rem">{theme.format_currency(revenue)}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="nx-kpi" style="min-height:96px"><div class="nx-kpi-label">Active rows</div><div class="nx-kpi-value" style="font-size:1.25rem">{theme.format_number(len(df))}</div></div>', unsafe_allow_html=True)

    # Revenue context — decluttered: weekly when >90 days + faint area, no spiky daily
    daily = df.groupby("date", as_index=False)["total_amount"].sum() if not df.empty else pd.DataFrame(columns=["date", "total_amount"])
    if not daily.empty:
        if len(daily) > 90:
            daily["wk"] = daily["date"].dt.to_period("W").dt.start_time
            plot_daily = daily.groupby("wk", as_index=False)["total_amount"].sum().rename(columns={"wk": "date"})
            title_suffix = " · weekly"
        else:
            plot_daily = daily
            title_suffix = ""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_daily["date"], y=plot_daily["total_amount"], line=dict(width=0), fill="tozeroy", fillcolor="rgba(59,130,246,0.05)", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=plot_daily["date"], y=plot_daily["total_amount"], mode="lines", line=dict(color=theme.ACCENT, width=1.8, shape="spline", smoothing=0.85), name="Revenue", hovertemplate="%{x|%b %d}<br>$%{y:,.0f}<extra></extra>"))
        fig.update_layout(title="Revenue context" + title_suffix, hovermode="x unified", margin=dict(t=36, b=16, l=12, r=12))
        fig.update_xaxes(nticks=6)
        fig.update_yaxes(nticks=4, tickprefix="$", separatethousands=True)
        theme.show_chart(fig, 220)

    if st.button("Generate executive report", type="primary", use_container_width=True):
        try:
            report = generate_executive_report()
            st.success(f"Report generated: {report}")
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")


st.markdown('<div class="nx-section">Assistant context</div>', unsafe_allow_html=True)
result_data = st.session_state.nexus_last_result
if isinstance(result_data, pd.DataFrame) and not result_data.empty:
    st.dataframe(result_data, use_container_width=True, hide_index=True, height=280)
else:
    st.dataframe(df.sort_values("datetime", ascending=False).head(20), use_container_width=True, hide_index=True, height=280)


# Payment method — decluttered: sorted bars, no extra legend, clean hover
if not df.empty:
    mix = df.groupby("payment_method", as_index=False)["total_amount"].sum().sort_values("total_amount")
else:
    mix = pd.DataFrame({"payment_method": [], "total_amount": []})
fig = go.Figure(go.Bar(x=mix["payment_method"], y=mix["total_amount"], marker_color=theme.COLORWAY[1], hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"))
fig.update_layout(title="Payment method context", margin=dict(t=36, b=24, l=12, r=12))
fig.update_xaxes(tickangle=0)
fig.update_yaxes(nticks=4, tickprefix="$", separatethousands=True)
theme.show_chart(fig, 260)


theme.footer("Lucciola v2.1.0 · Answers are based on available operational data.")
