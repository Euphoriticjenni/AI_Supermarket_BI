"""
Lucciola — Dithered Workflow OS v2.4
Single-page Streamlit analytics dashboard using the TypeUI Dithered design system.
Inspired by retro print halftone, monospace telemetry, and deep-navy canvas.

Run: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lucciola — Workflow OS v2.4 | Dithered",
    page_icon="⬢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Dithered Design System — Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Open+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --primary: #3B82F6;
    --secondary: #8B5CF6;
    --bg: #0F172A;
    --card-bg: #1E293B;
    --card-bg-hover: #22304a;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --border: rgba(59,130,246,0.30);
    --border-strong: rgba(59,130,246,0.50);
    --radius: 4px;
}

/* Base */
html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
}
.stApp {
    background: var(--bg);
    color: var(--text-primary);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
[data-testid="stHeader"] {
    background: var(--bg);
    border-bottom: 1px solid var(--border);
}
[data-testid="stSidebar"] {
    background: #0B1222;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}
footer, #MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }

/* Typography */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em;
    color: var(--text-primary) !important;
}
.mono { font-family: 'IBM Plex Mono', monospace !important; }

/* ── Header ── */
.dither-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 18px 22px;
    margin-bottom: 22px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    position: relative;
    overflow: hidden;
}
.dither-header::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: radial-gradient(rgba(59,130,246,0.18) 1px, transparent 0);
    background-size: 8px 8px;
    opacity: 0.55;
    pointer-events: none;
}
.dither-header-left {
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
    z-index: 1;
}
.dither-logo {
    width: 36px; height: 36px;
    background: var(--primary);
    border-radius: 4px;
    display: grid;
    place-items: center;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    color: white;
    font-size: 15px;
    letter-spacing: -0.02em;
    box-shadow: 0 0 0 1px rgba(59,130,246,0.4), 0 4px 16px rgba(59,130,246,0.30);
}
.dither-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 22px;
    line-height: 1;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.dither-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 3px;
}
.dither-header-right {
    display: flex;
    align-items: center;
    gap: 10px;
    position: relative;
    z-index: 1;
}
.badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 6px 10px;
    border-radius: 4px;
    border: 1px solid transparent;
    line-height: 1;
}
.badge-version {
    background: rgba(59,130,246,0.14);
    border-color: rgba(59,130,246,0.32);
    color: #93C5FD;
}
.badge-status {
    background: rgba(34,197,94,0.12);
    border-color: rgba(34,197,94,0.32);
    color: #86EFAC;
    display: inline-flex;
    align-items: center;
    gap: 7px;
}
.badge-status .dot {
    width: 7px; height: 7px;
    background: #22C55E;
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(34,197,94,0.9);
    animation: pulse 1.6s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(0.88); }
}

/* ── Dither Card ── */
.dither-card {
    position: relative;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 18px 16px;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s, background 0.2s;
}
.dither-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: radial-gradient(rgba(59,130,246,0.25) 1px, transparent 0);
    background-size: 8px 8px;
    opacity: 0.50;
    pointer-events: none;
}
.dither-card:hover {
    border-color: var(--border-strong);
    background: var(--card-bg-hover);
}
.dither-card > * { position: relative; z-index: 1; }
.dither-card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.dither-card-value {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 28px;
    line-height: 1;
    color: var(--text-primary);
    letter-spacing: -0.03em;
}
.dither-card-delta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    margin-top: 6px;
    letter-spacing: 0.02em;
}
.dither-card-delta.up { color: #86EFAC; }
.dither-card-delta.down { color: #FCA5A5; }
.dither-card-foot {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 10px;
    letter-spacing: 0.04em;
    border-top: 1px dashed rgba(148,163,184,0.18);
    padding-top: 8px;
}

/* ── Sections ── */
.section-head {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin: 22px 0 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-head::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] [role="columnheader"] {
    background: #0B1222 !important;
    color: #93C5FD !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] div[role="gridcell"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    border-bottom: 1px solid rgba(148,163,184,0.08) !important;
}

/* Sidebar widgets */
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}
.stSelectbox > div > div, .stSlider, [data-baseweb="select"] > div {
    border-radius: 4px !important;
}

/* Metric override hide default */
[data-testid="stMetric"] { display: none; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 8px; }

/* Halftone accent divider */
.halftone-divider {
    height: 1px;
    background-image: radial-gradient(rgba(139,92,246,0.35) 1px, transparent 0);
    background-size: 8px 8px;
    border: none;
    margin: 18px 0;
}

/* Tiny mono tag */
.mono-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_or_mock_data():
    """Try to load Lucciola transaction data; fallback to synthetic telemetry."""
    import pathlib
    root = pathlib.Path(__file__).parent
    candidates = [
        root / "data" / "processed" / "sales_features.parquet",
        root / "data" / "processed" / "sales_processed.parquet",
        root / "data" / "augmented" / "supermarket_sales.csv",
        root / "dashboard" / "data" / "sales_features.parquet",
    ]
    for p in candidates:
        if p.exists():
            try:
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p)
                # Normalize minimal columns for demo
                if "date" in df.columns or "Date" in df.columns:
                    date_col = "date" if "date" in df.columns else "Date"
                    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
                elif "datetime" in df.columns:
                    df["_date"] = pd.to_datetime(df["datetime"], errors="coerce")
                else:
                    df["_date"] = pd.to_datetime("2025-01-01")
                if "total_amount" in df.columns:
                    df["_value"] = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0)
                elif "Total" in df.columns:
                    df["_value"] = pd.to_numeric(df["Total"], errors="coerce").fillna(0)
                else:
                    df["_value"] = np.random.randint(80, 220, len(df))
                return df, True
            except Exception:
                continue
    # Synthetic fallback — Workflow OS telemetry
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=dt.date.today(), periods=60, freq="D")
    df = pd.DataFrame({
        "_date": np.repeat(dates, 20),
        "_value": rng.normal(180, 32, 60*20).clip(40, 320),
        "branch": rng.choice(["Downtown Central","Northgate Mall","Westside Plaza","East End Market","Southpark Express"], 60*20),
        "category": rng.choice(["Fresh Produce","Dairy & Eggs","Meat & Seafood","Bakery","Beverages","Pantry Staples"], 60*20),
    })
    return df, False

df_raw, is_real = load_or_mock_data()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — System Controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:6px 2px 14px;">
            <div style="width:28px;height:28px;background:#3B82F6;border-radius:4px;display:grid;place-items:center;color:white;font-family:'Space Grotesk',sans-serif;font-weight:700;">⬢</div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:13px;letter-spacing:-0.02em;color:#F8FAFC;">Lucciola OS</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:0.10em;text-transform:uppercase;color:#64748B;">Dithered • v2.4</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:rgba(59,130,246,0.20);margin:4px 0 14px;"></div>', unsafe_allow_html=True)

    st.markdown("**System Controls**")
    tier = st.selectbox("Operating Tier", ["All Tiers", "Tier-1 · Critical", "Tier-2 · Standard", "Tier-3 · Batch"], index=0)
    env = st.selectbox("Environment", ["Production", "Staging", "Canary", "Development"], index=0)
    auto_refresh = st.toggle("Auto-refresh (30s)", value=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    st.markdown("**Date Range**")
    today = dt.date.today()
    default_start = today - dt.timedelta(days=30)
    date_range = st.date_input("Date Range", value=(default_start, today))
    if isinstance(date_range, (tuple, list)) and len(date_range)==2:
        d0, d1 = date_range
    else:
        d0 = d1 = date_range if date_range else today

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    st.markdown("**Filters**")
    # Tier filter as slider (per spec)
    pending_cap = st.slider("Pending Cap", 0, 100, 40)
    sla_floor = st.slider("SLA Floor (%)", 80.0, 100.0, 98.5, step=0.1)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="background:#1E293B;border:1px solid rgba(59,130,246,0.30);border-radius:4px;padding:12px;position:relative;overflow:hidden;">
            <div style="position:absolute;inset:0;background-image:radial-gradient(rgba(59,130,246,0.22) 1px, transparent 0);background-size:8px 8px;opacity:0.45;"></div>
            <div style="position:relative;z-index:1;font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.08em;text-transform:uppercase;color:#64748B;">Operator</div>
            <div style="position:relative;z-index:1;display:flex;align-items:center;gap:10px;margin-top:8px;">
                <div style="width:30px;height:30px;border-radius:50%;background:#334155;display:grid;place-items:center;color:#F8FAFC;font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:11px;">AM</div>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:12px;color:#F8FAFC;">Alex Morgan</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#94A3B8;">Analyst · Tier-1 Access</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.caption("Workflow OS v2.4 • Dithered TypeUI")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dither-header">
  <div class="dither-header-left">
    <div class="dither-logo">◈</div>
    <div>
      <div class="dither-title">Workflow OS <span style="color:#93C5FD;">— Dithered</span></div>
      <div class="dither-subtitle">Lucciola &nbsp;·&nbsp; TypeUI Halftone System &nbsp;·&nbsp; Deep Navy Canvas</div>
    </div>
  </div>
  <div class="dither-header-right">
    <span class="badge badge-version">Workflow OS v2.4</span>
    <span class="badge badge-status"><span class="dot"></span> All Systems Nominal</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI Metrics Bar — 4 dither cards
# ─────────────────────────────────────────────────────────────────────────────
# Compute KPIs from data + synthetic overlays
if is_real and "_date" in df_raw.columns:
    # Filter by sidebar date range
    mask = (pd.to_datetime(df_raw["_date"]).dt.date >= d0) & (pd.to_datetime(df_raw["_date"]).dt.date <= d1)
    dff = df_raw.loc[mask]
    runs_today = int(len(dff[dff["_date"].dt.date == d1])) if len(dff) else 0
    # Simulate workflow KPIs derived from sales volume
    total_val = dff["_value"].sum() if len(dff) else 0
    success_rate = 97.2 + (total_val % 7) * 0.08  # stable pseudo
    pending_tasks = max(4, pending_cap - int(success_rate % 10))
    sla_track = min(99.9, max(sla_floor, 99.2 + (total_val % 5) * 0.05))
else:
    runs_today = 1284
    success_rate = 97.2
    pending_tasks = 12
    sla_track = 99.2

# Allow tier filter to slightly nudge numbers for interactivity
if tier != "All Tiers":
    runs_today = int(runs_today * (0.92 if "Batch" in tier else 1.06))
    success_rate = round(success_rate - 0.6 if "Batch" in tier else success_rate + 0.3, 1)
if env != "Production":
    pending_tasks += 3
    sla_track = round(sla_track - 0.4, 1)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="dither-card">
        <div class="dither-card-label">Runs Today</div>
        <div class="dither-card-value">{runs_today:,}</div>
        <div class="dither-card-delta up">▲ +8.4% vs yesterday</div>
        <div class="dither-card-foot">Updated {dt.datetime.now().strftime('%H:%M:%S')} • <span style="color:#3B82F6;">LIVE</span></div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="dither-card" style="border-left: 3px solid #8B5CF6;">
        <div class="dither-card-label">Success Rate</div>
        <div class="dither-card-value">{success_rate:.1f}<span style="font-size:16px;color:#94A3B8;">%</span></div>
        <div class="dither-card-delta up">▲ +0.6% 7-day trend</div>
        <div class="dither-card-foot">Target 98.0% • Halftone Violet</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="dither-card" style="border-left: 3px solid #F59E0B;">
        <div class="dither-card-label">Pending Tasks</div>
        <div class="dither-card-value">{pending_tasks}</div>
        <div class="dither-card-delta down">▼ -3 queue cleared</div>
        <div class="dither-card-foot">Cap {pending_cap} • Tier: {tier.split('·')[0].strip()}</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="dither-card" style="border-left: 3px solid #22C55E;">
        <div class="dither-card-label">SLA Track</div>
        <div class="dither-card-value">{sla_track:.1f}<span style="font-size:16px;color:#94A3B8;">%</span></div>
        <div class="dither-card-delta up">● Within threshold</div>
        <div class="dither-card-foot">Floor {sla_floor:.1f}% • {env}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main Section — 2 Columns
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.55, 1], gap="large")

# ── Left: Interactive Plotly chart ──
with left:
    st.markdown('<div class="section-head">Throughput Telemetry — Last 60 Days</div>', unsafe_allow_html=True)

    # Build time series: daily runs & success envelope
    if is_real:
        daily = df_raw.groupby(pd.to_datetime(df_raw["_date"]).dt.date).size().reset_index(name="runs")
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")
        # Smooth
        daily["ma7"] = daily["runs"].rolling(7, min_periods=1).mean()
        daily["ma30"] = daily["runs"].rolling(30, min_periods=1).mean()
        # Success band
        daily["succ_upper"] = daily["ma7"] * 1.08
        daily["succ_lower"] = daily["ma7"] * 0.92
    else:
        rng = np.random.default_rng(7)
        dates = pd.date_range(end=today, periods=60, freq="D")
        base = 18 + 4*np.sin(np.arange(60)/7) + rng.normal(0,1.2,60)
        daily = pd.DataFrame({
            "date": dates,
            "runs": (base*66).astype(int),
        })
        daily["ma7"] = daily["runs"].rolling(7, min_periods=1).mean()
        daily["ma30"] = daily["runs"].rolling(30, min_periods=1).mean()
        daily["succ_upper"] = daily["ma7"] * 1.07
        daily["succ_lower"] = daily["ma7"] * 0.93

    fig = go.Figure()
    # Confidence band
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["succ_upper"],
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["succ_lower"],
        fill="tonexty", fillcolor="rgba(139,92,246,0.10)",
        line=dict(width=0), name="Expected band", hoverinfo="skip"
    ))
    # MA30 dashed violet
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["ma30"],
        name="30-day avg", line=dict(color="#8B5CF6", width=1.6, dash="dash"),
        hovertemplate="30d avg: %{y:.0f}<extra></extra>"
    ))
    # MA7
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["ma7"],
        name="7-day avg", line=dict(color="#3B82F6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="Runs: %{y:.0f}<extra></extra>"
    ))
    # Dotted gridlines + deep background
    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=18, b=12),
        font=dict(family="IBM Plex Mono, monospace", color="#94A3B8", size=11),
        colorway=["#3B82F6","#8B5CF6","#22C55E"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11, color="#CBD5E1")),
        hovermode="x unified",
        xaxis=dict(
            showgrid=True, gridcolor="rgba(148,163,184,0.14)", gridwidth=1,
            griddash="dot", zeroline=False, linecolor="rgba(59,130,246,0.30)",
            tickfont=dict(color="#64748B", family="IBM Plex Mono", size=10),
            title=None,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(148,163,184,0.12)", gridwidth=1,
            griddash="dot", zeroline=False, linecolor="rgba(59,130,246,0.30)",
            tickfont=dict(color="#64748B", family="IBM Plex Mono", size=10),
            title=None,
        ),
        hoverlabel=dict(bgcolor="#1E293B", bordercolor="rgba(59,130,246,0.30)", font=dict(color="#F8FAFC", family="IBM Plex Mono")),
    )
    # Wrap in dither container via HTML + chart
    st.markdown("""
    <div class="dither-card" style="padding:12px 12px 8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span class="mono-tag">● Live Telemetry • #0F172A Canvas</span>
            <span class="mono-tag" style="color:#93C5FD;">BLUE / VIOLET ACCENTS</span>
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Mini footer stats under chart
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="mono-tag">Peak: <span style="color:#F8FAFC;">1,842 runs (Mar 12)</span></div>', unsafe_allow_html=True)
    c2.markdown('<div class="mono-tag">Avg: <span style="color:#F8FAFC;">1,284 / day</span></div>', unsafe_allow_html=True)
    c3.markdown('<div class="mono-tag" style="text-align:right;">Uptime: <span style="color:#86EFAC;">99.97%</span></div>', unsafe_allow_html=True)

# ── Right: Live Audit / Review Queue Table ──
with right:
    st.markdown('<div class="section-head">Live Audit — Review Queue</div>', unsafe_allow_html=True)

    # Build review queue table (monospace)
    audits = pd.DataFrame([
        ["#2847", "Stock Reconciliation", "Tier-1", "09:42:12", "OPEN", "HIGH"],
        ["#2846", "Refund Dispute — #8831", "Tier-2", "09:31:05", "IN REVIEW", "MED"],
        ["#2845", "SLA Breach Check", "Tier-1", "09:18:44", "OPEN", "HIGH"],
        ["#2844", "Inventory Threshold", "Tier-3", "09:02:19", "QUEUED", "LOW"],
        ["#2843", "Payment Reconciliation", "Tier-1", "08:55:33", "OPEN", "HIGH"],
        ["#2842", "Customer KYC Flag", "Tier-2", "08:41:07", "RESOLVED", "MED"],
        ["#2841", "Branch Transfer Log", "Tier-3", "08:29:51", "QUEUED", "LOW"],
        ["#2840", "Promo Code Anomaly", "Tier-2", "08:14:22", "IN REVIEW", "MED"],
    ], columns=["ID", "Workflow", "Tier", "Time", "Status", "Priority"])

    # Style with status colors via dataframe styling isn't reliable on dark canvas,
    # so we use a Plotly-free HTML table wrapped in dither-card for full monospace control.
    # Also provide an interactive st.dataframe for filtering.
    st.markdown("""
    <div class="dither-card" style="padding:14px 14px 10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.08em;text-transform:uppercase;color:#94A3B8;">Queue — 8 Items • Monospace Telemetry</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#3B82F6;">● LIVE</span>
        </div>
    """, unsafe_allow_html=True)

    # Use st.dataframe for interactivity but with custom CSS already handling monospace
    # Build styled display with emoji indicators
    def status_dot(s):
        return {"OPEN":"🔵 OPEN","IN REVIEW":"🟡 REVIEW","QUEUED":"⚪ QUEUED","RESOLVED":"🟢 DONE"}.get(s, s)

    display_df = audits.copy()
    display_df["Status"] = display_df["Status"].map(status_dot)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=320,
        column_config={
            "ID": st.column_config.TextColumn("ID", width="small"),
            "Workflow": st.column_config.TextColumn("Workflow", width="large"),
            "Tier": st.column_config.TextColumn("Tier", width="small"),
            "Time": st.column_config.TextColumn("Time", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Priority": st.column_config.TextColumn("Priority", width="small"),
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Action row
    a1, a2 = st.columns([1,1])
    a1.button("◉ Approve Batch", use_container_width=True, type="primary")
    a2.button("↻ Refresh Queue", use_container_width=True)

    st.markdown('<div class="halftone-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="dither-card" style="padding:14px;">
            <div class="mono-tag">System Note</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;line-height:1.6;color:#CBD5E1;margin-top:6px;">
                Dot grid is <span style="color:#3B82F6;">CSS-only</span> • No image assets. Cards use
                <span style="color:#8B5CF6;">1px dither</span> with 8px tiling. Text stays solid
                <span style="color:#F8FAFC;">#F8FAFC</span> for AAA contrast.
            </div>
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:22px;padding:14px 18px;border:1px solid rgba(59,130,246,0.20);border-radius:4px;background:#0B1222;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.06em;text-transform:uppercase;color:#64748B;">
    Lucciola • Dithered TypeUI • <span style="color:#94A3B8;">Workflow OS v2.4</span> • All Systems Nominal
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#475569;">
    Deep Navy #0F172A • Card #1E293B • Accent #3B82F6 / #8B5CF6
  </div>
</div>
""", unsafe_allow_html=True)
