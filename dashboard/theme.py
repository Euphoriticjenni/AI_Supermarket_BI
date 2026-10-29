"""Lucciola Enterprise Dark dashboard theme and shared UI helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence
import hashlib
import zlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

BG_ROOT = "#09090B"
BG_SURFACE = "#18181B"
BG_ELEVATED = "#27272A"
BORDER = "#27272A"
BORDER_FOCUS = "#3B82F6"
TEXT_PRIMARY = "#FAFAFA"
TEXT_SECONDARY = "#A1A1AA"
TEXT_MUTED = "#71717A"
ACCENT = "#3B82F6"
SUCCESS = "#22C55E"
WARNING = "#EAB308"
DANGER = "#EF4444"
COLORWAY = ["#3B82F6", "#8B5CF6", "#22C55E", "#EAB308", "#EF4444", "#06B6D4", "#EC4899", "#F97316"]
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{ color-scheme: dark; }}
html, body, [class*="css"], button, input, textarea, select {{ font-family: {FONT}; }}
body, p, li, span, label, div {{ line-height: 1.45; }}
h1, h2, h3 {{ color: {TEXT_PRIMARY}; font-weight: 600; letter-spacing: -0.025em; }}
.stApp {{ background: {BG_ROOT}; color: {TEXT_PRIMARY}; }}
.block-container {{ max-width: 1440px; padding: 32px 32px 48px; }}
[data-testid="stHeader"] {{ background: {BG_ROOT}; border-bottom: 1px solid {BORDER}; }}
[data-testid="stSidebar"] {{ width: 240px; min-width: 240px; background: {BG_ROOT}; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] .block-container {{ padding: 22px 14px; }}
[data-testid="stSidebar"] a {{ text-decoration: none !important; }}
.nx-brand {{ display:flex; align-items:center; gap:10px; padding:4px 10px 24px; }}
.nx-mark {{ display:grid; place-items:center; width:28px; height:28px; border-radius:6px; background:{ACCENT}; color:white; font-weight:700; }}
.nx-brand-name {{ color:{TEXT_PRIMARY}; font-size:1rem; font-weight:600; letter-spacing:-.02em; }}
.nx-brand-sub {{ color:{TEXT_MUTED}; font-size:.65rem; letter-spacing:.1em; text-transform:uppercase; margin-top:2px; }}
.nx-profile {{ border-top:1px solid {BORDER}; margin:24px 6px 0; padding:16px 4px 0; }}
.nx-profile-row {{ display:flex; align-items:center; gap:10px; }}
.nx-avatar {{ width:30px; height:30px; border-radius:50%; display:grid; place-items:center; background:{BG_ELEVATED}; color:{TEXT_PRIMARY}; font-size:.7rem; font-weight:600; }}
.nx-profile-name {{ color:{TEXT_PRIMARY}; font-size:.8rem; font-weight:600; }}
.nx-profile-role {{ color:{TEXT_MUTED}; font-size:.7rem; margin-top:2px; }}
.nx-topbar {{ display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid {BORDER}; padding:0 0 18px; margin-bottom:28px; }}
.nx-crumb {{ color:{TEXT_MUTED}; font-size:.78rem; }}
.nx-crumb strong {{ color:{TEXT_SECONDARY}; font-weight:500; }}
.nx-topbar-meta {{ color:{TEXT_MUTED}; font-size:.75rem; display:flex; align-items:center; gap:12px; }}
.nx-page-intro {{ margin-bottom:24px; }}
.nx-page-intro h1 {{ margin:0; font-size:1.65rem; }}
.nx-page-intro p {{ color:{TEXT_SECONDARY}; margin:6px 0 0; font-size:.9rem; }}
.nx-kpi {{ background:{BG_SURFACE}; border:1px solid {BORDER}; border-radius:8px; padding:16px 18px; box-shadow:0 1px 2px rgba(0,0,0,.3); min-height:132px; display:flex; flex-direction:column; justify-content:space-between; gap:4px; }}
.nx-kpi-label {{ color:{TEXT_MUTED}; font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; font-weight:600; }}
.nx-kpi-value {{ color:{TEXT_PRIMARY}; font-size:1.65rem; font-weight:700; font-variant-numeric:tabular-nums; letter-spacing:-.03em; line-height:1.15; margin:4px 0 2px; }}
.nx-kpi .stMetric {{ padding:0; margin:0; }}
.nx-kpi [data-testid="stMetric"] {{ gap:0; }}
.nx-kpi [data-testid="stMetricValue"] {{ color:{TEXT_PRIMARY}; font-size:1.60rem; font-weight:700; font-variant-numeric:tabular-nums; letter-spacing:-.03em; line-height:1.1; }}
.nx-kpi [data-testid="stMetricLabel"] {{ color:{TEXT_MUTED}; font-size:.68rem; letter-spacing:.05em; }}
.nx-kpi [data-testid="stMetricDelta"] {{ font-size:.72rem; margin-top:2px; }}
.nx-kpi svg.nx-spark {{ display:block; margin-top:8px; width:100%; opacity:.85; }}
.nx-kpi-row {{ display:flex; gap:16px; }}
.nx-kpi-row > div {{ flex:1; }}
.nx-section {{ color:{TEXT_PRIMARY}; font-size:1rem; font-weight:600; letter-spacing:-.015em; margin:26px 0 10px; }}
.nx-section-sub {{ color:{TEXT_SECONDARY}; font-size:.8rem; margin:-5px 0 12px; }}
.nx-panel {{ background:{BG_SURFACE}; border:1px solid {BORDER}; border-radius:8px; padding:20px; box-shadow:0 1px 2px rgba(0,0,0,.3); }}
.stButton > button {{ border-radius:6px; border:1px solid {BORDER}; background:transparent; color:{TEXT_SECONDARY}; font-weight:500; min-height:36px; }}
.stButton > button:hover {{ border-color:{BORDER_FOCUS}; color:{TEXT_PRIMARY}; background:{BG_ELEVATED}; }}
.stButton > button[kind="primary"] {{ color:#fff; background:{ACCENT}; border-color:{ACCENT}; }}
.stButton > button[kind="primary"]:hover {{ background:#2563EB; border-color:#2563EB; }}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap:24px; border-bottom:1px solid {BORDER}; }}
[data-testid="stTabs"] [data-baseweb="tab"] {{ color:{TEXT_MUTED}; padding:8px 4px; border-bottom:2px solid transparent; }}
[data-testid="stTabs"] [aria-selected="true"] {{ color:{TEXT_PRIMARY}; border-bottom-color:{ACCENT}; }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {{ background:{BG_ELEVATED}; border-color:{BORDER}; border-radius:6px; }}
[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:8px; overflow:hidden; }}
[data-testid="stDataFrame"] [role="columnheader"] {{ background:#111113; color:{TEXT_SECONDARY}; font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; }}
.stDataFrame {{ background:{BG_SURFACE}; }}
[data-testid="stMetric"] {{ font-variant-numeric:tabular-nums; }}
.stCaption, small {{ color:{TEXT_MUTED}; }}
hr {{ border-color:{BORDER}; }}
footer {{ visibility:hidden; }}
#MainMenu {{ visibility:hidden; }}
[data-testid="stToolbar"] {{ visibility:hidden; }}
[data-testid="stDecoration"] {{ display:none; }}
::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:{BG_ROOT}; }}
::-webkit-scrollbar-thumb {{ background:#3F3F46; border-radius:8px; }}
.nx-badge {{ display:inline-block; padding:3px 8px; border-radius:4px; font-size:.7rem; font-weight:600; letter-spacing:.04em; text-transform:uppercase; }}
.nx-badge-success {{ color:{SUCCESS}; background:rgba(34,197,94,.12); }}
.nx-badge-warning {{ color:{WARNING}; background:rgba(234,179,8,.12); }}
.nx-badge-danger {{ color:{DANGER}; background:rgba(239,68,68,.12); }}
.nx-badge-info {{ color:{ACCENT}; background:rgba(59,130,246,.12); }}
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def render_sidebar_nav(current_page: str) -> None:
    st.sidebar.markdown(
        '<div class="nx-brand"><div class="nx-mark">N</div><div><div class="nx-brand-name">Lucciola</div><div class="nx-brand-sub">Retail intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="nx-profile"><div class="nx-profile-row"><div class="nx-avatar">AM</div><div><div class="nx-profile-name">Alex Morgan</div><div class="nx-profile-role">Analyst · Pro plan</div></div></div></div>',
        unsafe_allow_html=True,
    )


def render_topbar(breadcrumbs: list[str]) -> None:
    crumb = " / ".join(breadcrumbs)
    st.markdown(
        f'<div class="nx-topbar"><div class="nx-crumb">Lucciola <strong>/ {crumb}</strong></div><div class="nx-topbar-meta">{pd.Timestamp.now():%b %d, %Y}<span class="nx-avatar">AM</span></div></div>',
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    delta: Optional[float] = None,
    direction: str = "up",
    icon: str = "",
    sparkline_data: Optional[Sequence[float]] = None,
) -> None:
    delta_value = None if delta is None else f"{delta:+.1f}%"
    with st.container():
        st.markdown(f'<div class="nx-kpi"><div class="nx-kpi-label">{label}</div>', unsafe_allow_html=True)
        st.metric(label="", value=value, delta=delta_value, delta_color="normal" if direction == "up" else "inverse")
        if sparkline_data is not None:
            st.markdown(sparkline_svg(sparkline_data), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_kpi_row(kpis: list[dict]) -> None:
    cols = st.columns(len(kpis))
    for col, item in zip(cols, kpis):
        with col:
            render_kpi_card(**item)


def render_section_header(title: str, subtitle: str = "", actions: Any = None) -> None:
    left, right = st.columns([4, 1])
    with left:
        st.markdown(f'<div class="nx-section">{title}</div>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<div class="nx-section-sub">{subtitle}</div>', unsafe_allow_html=True)
    if actions is not None:
        with right:
            actions()


def render_empty_state(message: str) -> None:
    st.markdown(
        f'<div class="nx-panel" style="text-align:center;color:{TEXT_SECONDARY};padding:42px">{message}</div>',
        unsafe_allow_html=True,
    )


def apply_chart_theme(fig: Any) -> Any:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT_SECONDARY, size=11),
        colorway=COLORWAY,
        margin=dict(l=12, r=12, t=36, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11, color=TEXT_SECONDARY)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=BG_ELEVATED, bordercolor=BORDER, font=dict(color=TEXT_PRIMARY, family=FONT, size=11)),
    )
    fig.update_xaxes(gridcolor="rgba(39,39,42,0.55)", zeroline=False, linecolor=BORDER, tickfont=dict(color=TEXT_MUTED, size=10), showgrid=True, nticks=8, tickangle=0)
    fig.update_yaxes(gridcolor="rgba(39,39,42,0.5)", zeroline=False, linecolor=BORDER, tickfont=dict(color=TEXT_MUTED, size=10), showgrid=True)
    return fig


def show_chart(fig: Any, height: int = 360) -> None:
    apply_chart_theme(fig)
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def format_currency(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.2f}"


def format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.0f}"


def format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1f}%"


def sparkline_svg(values: Sequence[float], color: str = ACCENT, width: int = 220, height: int = 32) -> str:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if len(clean) < 2:
        return ""
    lo, hi = min(clean), max(clean)
    span = hi - lo or 1.0
    # smooth with simple 3-pt average for visual clarity when many points
    if len(clean) > 14:
        clean = [clean[0]] + [(clean[i-1]+clean[i]+clean[i+1])/3 for i in range(1, len(clean)-1)] + [clean[-1]]
    points = " ".join(
        f"{i * width / (len(clean) - 1):.1f},{height - 3 - (v - lo) / span * (height - 6):.1f}"
        for i, v in enumerate(clean)
    )
    return f'<svg class="nx-spark" viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none"><polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/></svg>'


def _find(frame: pd.DataFrame, *names: str) -> Optional[str]:
    lookup = {str(c).lower(): str(c) for c in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _stable_customer(invoice: Any) -> str:
    return f"CUST-{zlib.crc32(str(invoice).encode()):08x}"[:13]


@st.cache_data(show_spinner="Loading Lucciola data")
def load_data() -> pd.DataFrame:
    candidates = [
        DATA_DIR / "processed" / "sales_features.parquet",
        DATA_DIR / "processed" / "sales_processed.parquet",
        DATA_DIR / "augmented" / "supermarket_sales.csv",
    ]
    frame = pd.DataFrame()
    for path in candidates:
        if path.exists():
            try:
                frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                break
            except Exception:
                continue
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "transaction_id",
                "datetime",
                "date",
                "customer_id",
                "product_id",
                "product_name",
                "category",
                "subcategory",
                "quantity",
                "unit_price",
                "total_amount",
                "profit",
                "payment_method",
                "branch",
                "membership_tier",
                "day_of_week",
                "hour_of_day",
                "is_return",
                "is_online_order",
            ]
        )
    aliases = {
        "transaction_id": ("transaction_id", "Invoice_ID"),
        "date": ("date", "Date"),
        "time": ("time", "Time"),
        "customer_id": ("customer_id", "Customer_ID"),
        "product_id": ("product_id", "Product_ID"),
        "product_name": ("product_name", "Product_Name"),
        "category": ("category", "Category", "Product_Line"),
        "subcategory": ("subcategory",),
        "quantity": ("quantity", "Quantity"),
        "unit_price": ("unit_price", "Unit_Price", "Selling_Price"),
        "total_amount": ("total_amount", "Total"),
        "profit": ("profit", "Profit", "Gross_Income"),
        "payment_method": ("payment_method", "Payment"),
        "branch": ("branch", "Branch"),
        "membership_tier": ("membership_tier", "Membership", "Customer_Type"),
        "day_of_week": ("day_of_week", "Day_Name"),
        "hour_of_day": ("hour_of_day", "Purchase_Hour"),
        "is_return": ("is_return",),
        "is_online_order": ("is_online_order",),
        "promotion_applied": ("promotion_applied",),
    }
    out = pd.DataFrame(index=frame.index)
    for target, names in aliases.items():
        source = _find(frame, *names)
        out[target] = frame[source].astype(object) if source else pd.NA
    out["transaction_id"] = out["transaction_id"].fillna(
        pd.Series(range(len(out)), index=out.index).map(lambda n: f"TXN-{n:08d}")
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    time_values = out["time"].astype("string") if "time" in out else pd.Series("00:00:00", index=out.index)
    out["datetime"] = pd.to_datetime(
        out["date"].dt.strftime("%Y-%m-%d") + " " + time_values.fillna("00:00:00"), errors="coerce"
    )
    out["date"] = out["datetime"].dt.normalize()
    out["customer_id"] = out["customer_id"].fillna(out["transaction_id"].map(_stable_customer))
    out["product_id"] = out["product_id"].fillna(out["product_name"]).fillna("UNKNOWN")
    out["product_name"] = out["product_name"].fillna(out["product_id"])
    out["subcategory"] = out["subcategory"].fillna("General")
    for col in ["quantity", "unit_price", "total_amount", "profit", "hour_of_day"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["payment_method"] = out["payment_method"].fillna("Unknown")
    out["branch"] = out["branch"].fillna("Unknown")
    out["category"] = out["category"].fillna("Unknown")
    out["membership_tier"] = out["membership_tier"].fillna("Unknown")
    out["day_of_week"] = out["day_of_week"].fillna(out["date"].dt.day_name())
    out["hour_of_day"] = out["hour_of_day"].astype(int)
    for col in ["is_return", "is_online_order", "promotion_applied"]:
        out[col] = out[col].fillna(False).astype(bool)
    return out.sort_values("datetime").reset_index(drop=True)


@st.cache_data(show_spinner="Loading product catalogue")
def load_products() -> pd.DataFrame:
    path = DATA_DIR / "augmented" / "products.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=["product_id", "name", "category", "unit_cost", "unit_price", "stock", "reorder_level"]
        )
    raw = pd.read_csv(path)
    mapping = {
        "product_id": ("product_id", "Product_ID"),
        "name": ("name", "Product_Name"),
        "category": ("category", "Category"),
        "unit_cost": ("unit_cost", "Cost"),
        "unit_price": ("unit_price", "Selling_Price"),
        "stock": ("stock", "Stock"),
        "reorder_level": ("reorder_level", "Reorder_Level"),
    }
    out = pd.DataFrame(index=raw.index)
    for target, aliases in mapping.items():
        source = _find(raw, *aliases)
        out[target] = raw[source].astype(object) if source else 0
    out["stock"] = pd.to_numeric(out["stock"], errors="coerce").fillna(100)
    out["reorder_level"] = pd.to_numeric(out["reorder_level"], errors="coerce").fillna(20)
    return out


def date_filter(frame: pd.DataFrame, label: str = "Date range") -> pd.DataFrame:
    if frame.empty:
        return frame
    minimum, maximum = frame["date"].min().date(), frame["date"].max().date()
    chosen = st.date_input(label, value=(minimum, maximum), min_value=minimum, max_value=maximum)
    if isinstance(chosen, tuple) and len(chosen) == 2:
        start, end = chosen
    else:
        start = end = chosen if chosen else minimum
    return frame[(frame["date"].dt.date >= start) & (frame["date"].dt.date <= end)]


# Compatibility aliases used by the existing dashboard helpers.
def apply_theme() -> None:
    inject_theme()


def render_nav(active: str = "Overview") -> None:
    render_sidebar_nav(active)


def currency(value: Any, compact: bool = True) -> str:
    return format_currency(value) if compact else (
        "—" if value is None or pd.isna(value) else f"${float(value):,.2f}"
    )


def fmt_int(value: Any) -> str:
    return format_number(value)


def pct_change(current: Any, previous: Any) -> Optional[float]:
    if current is None or previous in (None, 0) or pd.isna(current) or pd.isna(previous):
        return None
    return (float(current) - float(previous)) / float(previous) * 100


def kpi_cards(cards: Sequence[dict]) -> None:
    render_kpi_row(
        [
            {
                "label": c.get("label", ""),
                "value": c.get("value", "—"),
                "delta": c.get("delta"),
                "direction": "up" if (c.get("delta") or 0) >= 0 else "down",
                "sparkline_data": c.get("spark"),
            }
            for c in cards
        ]
    )


def chart(fig: Any, height: Optional[int] = None, **kwargs: Any) -> None:
    if kwargs.get("title"):
        fig.update_layout(title=kwargs["title"])
    show_chart(fig, height or 360)


def restyle_series(fig: Any, palette: Optional[Sequence[str]] = None) -> None:
    colors = list(palette or COLORWAY)
    for i, trace in enumerate(fig.data):
        if trace.type in {"bar", "scatter", "histogram"}:
            if trace.type == "scatter":
                trace.line.color = colors[i % len(colors)]
            else:
                trace.marker.color = colors[i % len(colors)]


def inline_badge(text: str, tone: str = "info") -> str:
    return f'<span class="nx-badge nx-badge-{tone}">{text}</span>'


def footer(note: str = "Lucciola v2.1.0") -> None:
    st.markdown(
        f'<div style="border-top:1px solid {BORDER};margin-top:32px;padding-top:16px;color:{TEXT_MUTED};font-size:.72rem">{note}</div>',
        unsafe_allow_html=True,
    )