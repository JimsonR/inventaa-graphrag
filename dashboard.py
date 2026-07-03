"""
Inventaa Lighting Catalog & Knowledge Base — Streamlit Dashboard
Run:  streamlit run dashboard.py
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inventaa · Lighting KB",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
C_BG       = "#0d1117"
C_SURFACE  = "#161b22"
C_SURFACE2 = "#1c2128"
C_BORDER   = "#30363d"
C_BORDER2  = "#3d444d"
C_TEXT     = "#e6edf3"
C_MUTED    = "#8b949e"
C_BLUE     = "#58a6ff"
C_GREEN    = "#3fb950"
C_AMBER    = "#e3b341"
C_PURPLE   = "#d2a8ff"
C_RED      = "#f85149"

CHART_BG   = C_BG
PAPER_BG   = C_SURFACE
GRID_COLOR = C_BORDER
FONT_COLOR = C_TEXT

CHART_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=CHART_BG,
    font=dict(color=FONT_COLOR, family="Inter, Segoe UI, sans-serif", size=12),
    margin=dict(t=30, b=20, l=20, r=20),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=C_MUTED),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=C_MUTED),
)

CAT_COLORS = {
    "Gate & Pillar Lights": C_AMBER,
    "Indoor & Ceiling Lights": C_BLUE,
    "Solar Lights": C_GREEN,
    "Outdoor Wall Lights": C_PURPLE,
    "Bollard & Garden Lights": "#39c5bb",
    "Divine & Temple Lights": "#ff7b72",
    "Street Lights": "#79c0ff",
    "Pathway & Step Lights": "#d2a8ff",
    "General Purpose Lights": "#ffa657",
    "Flood Lights": "#ff7b72",
    "Panel Lights": "#58a6ff",
    "Bulkhead Lights": "#8b949e",
    "Outdoor Commercial Lights": "#3fb950",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}

[data-testid="stAppViewContainer"]  {{ background: {C_BG}; }}
[data-testid="stSidebar"]           {{ background: {C_SURFACE}; border-right: 1px solid {C_BORDER}; }}
[data-testid="stSidebarNav"]        {{ display: none; }}
[data-testid="stHeader"]            {{ background: transparent; }}
[data-testid="stToolbar"]           {{ display: none; }}
footer                              {{ display: none; }}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    transition: border-color .2s;
}}
[data-testid="metric-container"]:hover {{ border-color: {C_BLUE}; }}
[data-testid="metric-container"] label {{
    color: {C_MUTED} !important;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .05em;
}}
[data-testid="stMetricValue"] {{
    color: {C_TEXT} !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
}}
[data-testid="stMetricDelta"] {{ font-size: 0.78rem !important; }}

/* ── Section header ── */
.sh {{
    display: flex; align-items: center; gap: .55rem;
    font-size: 1rem; font-weight: 600; color: {C_TEXT};
    margin: 1.6rem 0 .8rem;
}}
.sh::before {{
    content: ''; display: inline-block;
    width: 3px; height: 1.1em;
    background: {C_BLUE}; border-radius: 2px;
}}

/* ── Page hero ── */
.hero {{
    background: linear-gradient(135deg, {C_SURFACE} 0%, #0d1f3c 100%);
    border: 1px solid {C_BORDER};
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}}
.hero h1 {{ color: {C_TEXT}; font-size: 1.8rem; font-weight: 700; margin: 0 0 .3rem; }}
.hero p  {{ color: {C_MUTED}; font-size: .9rem; margin: 0; }}
.hero-badge {{
    display: inline-block; background: rgba(88,166,255,.12);
    color: {C_BLUE}; border: 1px solid rgba(88,166,255,.3);
    border-radius: 20px; font-size: .72rem; font-weight: 600;
    padding: 3px 10px; margin-right: 6px;
}}

/* ── Card grid ── */
.card-grid-3 {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .75rem;
    align-items: start;
    margin-top: .5rem;
}}
.card-grid-2 {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: .75rem;
    align-items: start;
    margin-top: .5rem;
}}
@media (max-width: 900px) {{
    .card-grid-3, .card-grid-2 {{ grid-template-columns: 1fr 1fr; }}
}}
@media (max-width: 600px) {{
    .card-grid-3, .card-grid-2 {{ grid-template-columns: 1fr; }}
}}

/* ── Product card (equivalent to Doctor card) ── */
.pcard {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 14px;
    padding: 1rem 1.1rem;
    transition: border-color .2s, box-shadow .2s;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    gap: .45rem;
}}
.pcard::before {{
    content: ''; position: absolute; top:0; left:0; right:0;
    height: 2px;
    background: linear-gradient(90deg, {C_AMBER}, {C_BLUE});
    opacity: 0;
    transition: opacity .2s;
}}
.pcard:hover {{ border-color: {C_AMBER}; box-shadow: 0 4px 20px rgba(227,179,65,.08); }}
.pcard:hover::before {{ opacity: 1; }}

.avatar {{
    width: 40px; height: 40px; border-radius: 50%;
    background: linear-gradient(135deg, #2b2208, #1c1808);
    border: 1.5px solid {C_AMBER};
    display: flex; align-items: center; justify-content: center;
    font-size: .85rem; font-weight: 700; color: {C_AMBER};
    flex-shrink: 0;
}}
.pcard-top {{ display: flex; align-items: center; gap: .6rem; }}
.pcard-info {{ flex: 1; min-width: 0; }}
.pname  {{ font-size: .9rem; font-weight: 600; color: {C_TEXT}; margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.psku {{ font-size: .7rem; color: {C_AMBER}; margin: .1rem 0 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.pspec  {{ font-size: .75rem; color: {C_MUTED}; margin: 0; }}

.badge {{
    display: inline-flex; align-items: center; gap: 4px;
    border-radius: 20px; font-size: .68rem; font-weight: 600;
    padding: 2px 9px; margin: 2px 3px 0 0;
}}
.badge-default {{ background: {C_SURFACE2}; color: {C_MUTED}; border: 1px solid {C_BORDER}; }}
.badge-blue    {{ background: rgba(88,166,255,.1); color: {C_BLUE}; border: 1px solid rgba(88,166,255,.25); }}
.badge-green   {{ background: rgba(63,185,80,.1); color: {C_GREEN}; border: 1px solid rgba(63,185,80,.25); }}
.badge-amber   {{ background: rgba(227,179,65,.1); color: {C_AMBER}; border: 1px solid rgba(227,179,65,.25); }}
.badge-purple  {{ background: rgba(210,168,255,.1); color: {C_PURPLE}; border: 1px solid rgba(210,168,255,.25); }}

.avail-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    display: inline-block; margin-right: 4px;
}}
.dot-green {{ background: {C_GREEN}; box-shadow: 0 0 6px {C_GREEN}; }}
.dot-amber {{ background: {C_AMBER}; box-shadow: 0 0 6px {C_AMBER}; }}
.dot-red   {{ background: {C_MUTED}; }}

.view-btn {{
    display: inline-block; margin-top: .65rem;
    background: linear-gradient(135deg, #1f4277, #132c52);
    color: #fff !important;
    border-radius: 8px; padding: 5px 14px;
    font-size: .75rem; font-weight: 600;
    text-decoration: none; transition: opacity .15s;
}}
.view-btn:hover {{ opacity: .88; }}

/* ── Category / Collection card (equivalent to Hospital card) ── */
.ccard {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    margin-bottom: .75rem;
    transition: border-color .2s;
}}
.ccard:hover {{ border-color: {C_AMBER}; }}
.ctitle {{ font-size: 1rem; font-weight: 600; color: {C_TEXT}; margin: 0 0 .2rem; }}
.cloc   {{ font-size: .78rem; color: {C_MUTED}; margin: 0 0 .6rem; }}
.cbar   {{
    background: {C_BORDER}; border-radius: 4px; height: 6px; margin: .5rem 0 .3rem;
}}
.cbar-fill {{
    height: 6px; border-radius: 4px;
    background: linear-gradient(90deg, {C_AMBER}, {C_BLUE});
    transition: width .4s;
}}

/* ── Sidebar nav ── */
.nav-item {{
    display: flex; align-items: center; gap: .6rem;
    padding: .5rem .75rem; border-radius: 8px;
    font-size: .875rem; color: {C_MUTED};
    cursor: pointer; margin: 2px 0;
    transition: background .15s, color .15s;
}}
.nav-item:hover {{ background: {C_SURFACE2}; color: {C_TEXT}; }}
.nav-item.active {{ background: rgba(88,166,255,.12); color: {C_BLUE}; font-weight: 600; }}

/* ── Stat chip in sidebar ── */
.stat-chip {{
    display: flex; justify-content: space-between; align-items: center;
    background: {C_SURFACE2}; border: 1px solid {C_BORDER};
    border-radius: 10px; padding: .45rem .75rem; margin: .35rem 0;
    font-size: .8rem;
}}
.stat-chip span:first-child {{ color: {C_MUTED}; }}
.stat-chip span:last-child  {{ color: {C_TEXT}; font-weight: 600; }}

/* ── Divider ── */
hr {{ border-color: {C_BORDER} !important; margin: .75rem 0 !important; }}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {C_SURFACE} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 12px !important;
    margin-bottom: .5rem;
}}
[data-testid="stExpander"] summary {{
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    font-size: .875rem; color: {C_TEXT}; padding: .6rem 1rem;
}}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select {{
    background: {C_SURFACE2} !important;
    border: 1px solid {C_BORDER} !important;
    color: {C_TEXT} !important;
    border-radius: 8px !important;
}}
</style>
""", unsafe_allow_html=True)


# ── DB helpers ─────────────────────────────────────────────────────────────────
_DB_PATH = "data/db/inventaa_knowledge_base.db"

def _conn():
    if not Path(_DB_PATH).exists():
        return None
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_products() -> pd.DataFrame:
    conn = _conn()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query("""
            SELECT id, sku, name, price_num, regular_price, discount_percentage,
                   rating_score, review_count, image_url, url, description,
                   feature_descriptions, has_variants, wattage, tenant,
                   categories, features, use_cases, color_options, wattage_options
            FROM products
            ORDER BY name
        """, conn)

        str_cols = ["regular_price", "image_url", "url", "description", "feature_descriptions",
                    "categories", "features", "use_cases", "color_options", "wattage_options"]
        for c in str_cols:
            if c in df.columns:
                df[c] = df[c].fillna("")

        df["primary_category"] = df["categories"].str.split(",").str[0].str.strip()
        df["primary_category"] = df["primary_category"].replace("", "General Purpose Lights")

        df["has_discount"] = (df["discount_percentage"].fillna(0) > 0)
        return df
    except Exception as exc:
        st.error(f"DB error: {exc}")
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data(ttl=300)
def fetch_stats() -> dict:
    conn = _conn()
    if conn is None:
        return {}
    try:
        cur = conn.cursor()
        out = {}
        out["total_products"] = cur.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        out["total_specs"]    = cur.execute("SELECT COUNT(*) FROM product_specs").fetchone()[0]
        out["total_variants"] = cur.execute("SELECT COUNT(*) FROM product_variants").fetchone()[0]
        out["avg_price"]      = cur.execute("SELECT COALESCE(AVG(price_num),0) FROM products WHERE price_num > 0").fetchone()[0]
        out["avg_rating"]     = cur.execute("SELECT COALESCE(AVG(rating_score),0) FROM products WHERE rating_score > 0").fetchone()[0]
        out["max_discount"]   = cur.execute("SELECT COALESCE(MAX(discount_percentage),0) FROM products").fetchone()[0]
        out["on_sale"]        = cur.execute("SELECT COUNT(*) FROM products WHERE discount_percentage > 0").fetchone()[0]
        return out
    finally:
        conn.close()


@st.cache_data(ttl=300)
def fetch_categories() -> pd.DataFrame:
    conn = _conn()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query("""
            SELECT p.sku, p.name, p.price_num, p.rating_score, p.categories, p.features, p.variants_count
            FROM (
                SELECT sku, name, price_num, rating_score, categories, features,
                       (SELECT COUNT(*) FROM product_variants pv WHERE pv.product_sku = products.sku) AS variants_count
                FROM products
            ) p
        """, conn)
        
        # Expand comma-separated categories
        expanded = []
        for _, row in df.iterrows():
            cats = [c.strip() for c in (row["categories"] or "").split(",") if c.strip()]
            if not cats:
                cats = ["General Purpose Lights"]
            for c in cats:
                expanded.append({
                    "category": c,
                    "sku": row["sku"],
                    "price_num": row["price_num"],
                    "rating_score": row["rating_score"],
                    "variants_count": row["variants_count"],
                    "features": row["features"]
                })
        
        if not expanded:
            return pd.DataFrame()
            
        exp_df = pd.DataFrame(expanded)
        grouped = exp_df.groupby("category").agg(
            product_count=("sku", "count"),
            avg_price=("price_num", lambda x: int(x[x > 0].mean()) if any(x > 0) else 0),
            max_price=("price_num", "max"),
            min_price=("price_num", lambda x: int(x[x > 0].min()) if any(x > 0) else 0),
            avg_rating=("rating_score", lambda x: round(x[x > 0].mean(), 2) if any(x > 0) else 0.0),
            total_variants=("variants_count", "sum")
        ).reset_index().sort_values("product_count", ascending=False)
        return grouped
    except Exception as exc:
        st.error(f"Category load error: {exc}")
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data(ttl=300)
def fetch_specs() -> pd.DataFrame:
    conn = _conn()
    if conn is None:
        return pd.DataFrame(columns=["product_sku", "product_name", "categories", "spec_key", "spec_value"])
    try:
        df = pd.read_sql_query("""
            SELECT ps.product_sku, p.name AS product_name, p.categories, ps.spec_key, ps.spec_value
            FROM product_specs ps
            LEFT JOIN products p ON p.sku = ps.product_sku
            ORDER BY p.name, ps.spec_key
        """, conn)
        for c in ["product_name", "categories", "spec_key", "spec_value"]:
            if c in df.columns:
                df[c] = df[c].fillna("")
        return df
    except Exception as exc:
        st.error(f"Specs load error: {exc}")
        return pd.DataFrame(columns=["product_sku", "product_name", "categories", "spec_key", "spec_value"])
    finally:
        conn.close()


@st.cache_data(ttl=300)
def fetch_variants() -> pd.DataFrame:
    conn = _conn()
    if conn is None:
        return pd.DataFrame(columns=["product_sku", "product_name", "variant_sku", "color_option", "wattage_option", "price"])
    try:
        df = pd.read_sql_query("""
            SELECT pv.product_sku, p.name AS product_name, pv.variant_sku,
                   COALESCE(pv.color_option, '') AS color_option,
                   COALESCE(pv.wattage_option, '') AS wattage_option,
                   COALESCE(pv.price_num, p.price_num) AS price
            FROM product_variants pv
            LEFT JOIN products p ON p.sku = pv.product_sku
            ORDER BY p.name, pv.variant_sku
        """, conn)
        return df
    except Exception as exc:
        return pd.DataFrame(columns=["product_sku", "product_name", "variant_sku", "color_option", "wattage_option", "price"])
    finally:
        conn.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def cat_color(cat: str) -> str:
    return CAT_COLORS.get(cat, C_BLUE)

def price_badge(price) -> str:
    n = int(price) if pd.notna(price) and price > 0 else 0
    if n > 0:
        return f'<span class="badge badge-amber"><span class="avail-dot dot-amber"></span>Rs. {n:,}</span>'
    return f'<span class="badge badge-default"><span class="avail-dot dot-red"></span>Price on Req</span>'

def rating_badge(rating) -> str:
    r = float(rating) if pd.notna(rating) and rating > 0 else 0.0
    if r > 0:
        return f'<span class="badge badge-green">★ {r:.1f}</span>'
    return f'<span class="badge badge-default">★ Unrated</span>'


# ── Load primary data ─────────────────────────────────────────────────────────

df          = fetch_products()
stats       = fetch_stats()
cat_df      = fetch_categories()
specs_df    = fetch_specs()
variants_df = fetch_variants()

all_categories = sorted(cat_df["category"].unique()) if not cat_df.empty else []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="padding:.75rem 0 .5rem">
      <div style="font-size:1.4rem;font-weight:800;color:{C_TEXT};letter-spacing:-.02em">
        💡 Inventaa
      </div>
      <div style="font-size:.75rem;color:{C_MUTED};margin-top:.1rem">Lighting Knowledge Base</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio("nav", [
        "📊  Overview",
        "📂  Categories & Collections",
        "💡  Product Directory",
        "⚙️  Specifications & Variants",
        "🕸️  Knowledge Graph",
        "🔍  Category & Product Explorer",
    ], label_visibility="collapsed")

    st.divider()

    n_prod  = stats.get("total_products", len(df))
    n_cat   = len(cat_df)
    n_var   = stats.get("total_variants", 0)
    n_spec  = stats.get("total_specs", 0)

    for label, val in [
        ("Products",    f"{n_prod:,}"),
        ("Categories",  str(n_cat)),
        ("Variants",    f"{n_var:,}"),
        ("Specs Indexed", f"{n_spec:,}"),
    ]:
        st.markdown(f"""
        <div class="stat-chip">
          <span>{label}</span><span>{val}</span>
        </div>""", unsafe_allow_html=True)

    st.divider()
    db_ok = Path(_DB_PATH).exists()
    st.markdown(f"""
    <div style="font-size:.75rem;color:{C_MUTED}">
      <div style="margin-bottom:.3rem;font-weight:600;color:{C_TEXT}">Data Sources</div>
      <div style="margin:.25rem 0">{'🟢' if db_ok else '🔴'} SQLite · Tri-Store Catalog</div>
      <div style="margin:.25rem 0">🟢 Neo4j · Semantic Graph Index</div>
      <div style="margin:.25rem 0">🟢 Pinecone · Dense Vector Embeddings</div>
    </div>
    """, unsafe_allow_html=True)


if df.empty:
    st.error("No data found in SQLite catalog. Ensure `inventaa_knowledge_base.db` is populated.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊  Overview":
    st.markdown(f"""
    <div class="hero">
      <h1>📊 Inventaa Knowledge Base Overview</h1>
      <p>Real-time telemetry and analytics across the authoritative Inventaa LED lighting catalog.</p>
      <div style="margin-top:.75rem">
        {"".join(f'<span class="hero-badge">{c}</span>' for c in all_categories[:6])}
        {"<span class='hero-badge'>+ more</span>" if len(all_categories) > 6 else ""}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row 1 ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Products", f"{n_prod:,}")
    c2.metric("Categories",     str(n_cat))
    c3.metric("Total Variants", f"{n_var:,}")
    c4.metric("Avg Sale Price", f"Rs. {int(stats.get('avg_price', 0)):,}")
    c5.metric("Avg Star Rating", f"★ {stats.get('avg_rating', 0.0):.2f}")

    # ── KPI row 2 ─────────────────────────────────────────────────────────────
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Specs Indexed",  f"{n_spec:,}")
    s2.metric("On Sale",        f"{stats.get('on_sale', 0):,}")
    s3.metric("Max Discount",   f"{stats.get('max_discount', 0)}%")
    s4.metric("With Variants",  f"{df['has_variants'].sum():,}")
    s5.metric("Wattage Options", str(df['wattage_options'].replace('', pd.NA).dropna().nunique()))

    st.markdown('<div class="sh">Category & Collection Network</div>', unsafe_allow_html=True)

    max_prods = cat_df["product_count"].max() if not cat_df.empty else 1
    cat_mini_html = ""
    for _, h in cat_df.iterrows():
        pct = int(h["product_count"] / max_prods * 100) if max_prods else 0
        cc  = cat_color(h["category"])
        cat_mini_html += f"""
<div class="ccard" style="border-top:2px solid {cc};margin:0">
  <div class="ctitle">{h['category']}</div>
  <div class="cloc">🏷️ Avg: Rs. {h['avg_price']:,} · ★ {h['avg_rating']:.1f}</div>
  <div class="cbar"><div class="cbar-fill" style="width:{pct}%;background:{cc}"></div></div>
  <div style="display:flex;justify-content:space-between;font-size:.78rem">
    <span style="color:{cc};font-weight:600">{int(h['product_count'])} products</span>
    <span style="color:{C_MUTED}">{int(h.get('total_variants',0)):,} variants</span>
  </div>
</div>"""
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.65rem;align-items:start">'
        f'{cat_mini_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="sh">Products by Category</div>', unsafe_allow_html=True)
        sc = cat_df.copy()
        fig = px.bar(sc.sort_values("product_count"), x="product_count", y="category",
                     orientation="h", color="product_count", color_continuous_scale="Blues", text="product_count",
                     labels={"product_count": "Product Count", "category": "Category"})
        fig.update_traces(textposition="outside", textfont_size=11)
        fig.update_coloraxes(showscale=False)
        fig.update_layout(**CHART_LAYOUT, height=460)
        st.plotly_chart(fig, width='stretch')

    with col_r:
        st.markdown('<div class="sh">Catalog Price Distribution</div>', unsafe_allow_html=True)
        price_df = df[df["price_num"] > 0]
        fig2 = px.histogram(price_df, x="price_num", nbins=15,
                            color_discrete_sequence=[C_AMBER],
                            labels={"price_num": "Price (INR)", "count": "Products"})
        fig2.update_layout(**CHART_LAYOUT, height=220, bargap=0.06)
        fig2.update_traces(marker_line_color=C_BORDER, marker_line_width=1)
        st.plotly_chart(fig2, width='stretch')

        st.markdown('<div class="sh">Wattage Distribution</div>', unsafe_allow_html=True)
        w_df = df[df["wattage"] > 0]
        fig_w = px.histogram(w_df, x="wattage", nbins=12,
                             color_discrete_sequence=[C_BLUE],
                             labels={"wattage": "Wattage (W)", "count": "Products"})
        fig_w.update_layout(**CHART_LAYOUT, height=180, bargap=0.06)
        fig_w.update_traces(marker_line_color=C_BORDER, marker_line_width=1)
        st.plotly_chart(fig_w, width='stretch')

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    col_l2, col_r2 = st.columns([3, 2])

    with col_l2:
        st.markdown('<div class="sh">Top Rated Lighting Fixtures</div>', unsafe_allow_html=True)
        top_r = df[df["rating_score"] > 0].sort_values(["rating_score", "review_count"], ascending=[False, False]).head(10)
        fig3 = px.bar(top_r, x="rating_score", y="name", orientation="h",
                      color="review_count", color_continuous_scale="YlOrBr", text="rating_score",
                      labels={"rating_score": "Star Rating", "name": "Product Name", "review_count": "Reviews"})
        fig3.update_traces(texttemplate="%{text:.2f} ★", textposition="outside", textfont_size=11)
        fig3.update_coloraxes(showscale=False)
        fig3.update_layout(**CHART_LAYOUT, height=320)
        fig3.update_yaxes(autorange="reversed")
        st.plotly_chart(fig3, width='stretch')

    with col_r2:
        st.markdown('<div class="sh">Top Features Indexed</div>', unsafe_allow_html=True)
        feats_s = df["features"].str.split(",").explode().str.strip()
        feats_s = feats_s[feats_s != ""].value_counts().head(8).reset_index()
        feats_s.columns = ["Feature", "Count"]
        fig4 = px.pie(feats_s, values="Count", names="Feature", hole=0.55,
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig4.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
        fig4.update_layout(**CHART_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig4, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CATEGORIES & COLLECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📂  Categories & Collections":
    st.markdown(f"""
    <div class="hero">
      <h1>📂 Categories & Collections Directory</h1>
      <p>Explore authoritative metrics across Inventaa's architectural, outdoor, and domestic collections.</p>
    </div>
    """, unsafe_allow_html=True)

    fcol1, fcol2 = st.columns([2, 1])
    with fcol1: search_cat = st.text_input("🔍 Search collection name...", "")
    with fcol2: sort_by  = st.selectbox("Sort by", ["Product Count (High → Low)", "Avg Price (High → Low)", "Avg Price (Low → High)", "Avg Rating"])

    cdf = cat_df.copy()
    if search_cat:
        cdf = cdf[cdf["category"].str.contains(search_cat, case=False, na=False)]

    if sort_by == "Avg Price (High → Low)":
        cdf = cdf.sort_values("avg_price", ascending=False)
    elif sort_by == "Avg Price (Low → High)":
        cdf = cdf.sort_values("avg_price", ascending=True)
    elif sort_by == "Avg Rating":
        cdf = cdf.sort_values("avg_rating", ascending=False)
    else:
        cdf = cdf.sort_values("product_count", ascending=False)

    max_p = cdf["product_count"].max() if not cdf.empty else 1
    cols  = st.columns(3)
    for i, (_, row) in enumerate(cdf.iterrows()):
        cc = cat_color(row["category"])
        pct = int(row["product_count"] / max_p * 100) if max_p else 0
        with cols[i % 3]:
            st.markdown(f"""
<div class="ccard" style="border-left:3px solid {cc}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <div class="ctitle">{row['category']}</div>
      <div class="cloc">🏷️ Price Range: Rs. {row['min_price']:,} – {row['max_price']:,}</div>
    </div>
    <span class="badge badge-amber" style="margin:0;font-size:.75rem">★ {row['avg_rating']:.2f}</span>
  </div>
  <div class="cbar"><div class="cbar-fill" style="width:{pct}%;background:{cc}"></div></div>
  <div style="display:flex;justify-content:space-between;font-size:.78rem;margin-top:.4rem">
    <span style="color:{cc};font-weight:600">💡 {row['product_count']} Products</span>
    <span style="color:{C_MUTED}">⚙️ {row['total_variants']} Variants</span>
  </div>
</div>""", unsafe_allow_html=True)

            with st.expander(f"View {row['product_count']} products in collection"):
                sub_prods = df[df["categories"].str.contains(row["category"], case=False, na=False)]
                for _, pr in sub_prods.iterrows():
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;padding:.35rem 0;border-bottom:1px solid {C_BORDER};font-size:.8rem">
                      <div style="font-weight:500;color:{C_TEXT}">{pr['name']} <span style="color:{C_MUTED}">({pr['sku']})</span></div>
                      <div>{price_badge(pr['price_num'])}</div>
                    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRODUCT DIRECTORY
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "💡  Product Directory":
    st.markdown(f"""
    <div class="hero">
      <h1>💡 Master Lighting Catalog</h1>
      <p>Filter, search, and inspect all indexed lighting fixtures in the Inventaa catalog.</p>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([2, 2, 2, 3])
    with f1: sel_cat   = st.selectbox("Category Filter", ["All"] + all_categories)
    with f2: sel_use   = st.selectbox("Use Case", ["All"] + sorted(list(set(df["use_cases"].str.split(",").explode().str.strip().replace('', pd.NA).dropna()))))
    with f3: max_price = st.slider("Max Price (INR)", 0, int(df["price_num"].max() or 10000), int(df["price_num"].max() or 10000), step=500)
    with f4: search_q  = st.text_input("🔍 Search Name, SKU, or Feature...", "")

    sub = df.copy()
    if sel_cat != "All":
        sub = sub[sub["categories"].str.contains(sel_cat, case=False, na=False)]
    if sel_use != "All":
        sub = sub[sub["use_cases"].str.contains(sel_use, case=False, na=False)]
    if max_price > 0:
        sub = sub[sub["price_num"] <= max_price]
    if search_q:
        q = search_q.lower()
        sub = sub[sub["name"].str.lower().str.contains(q, na=False) |
                  sub["sku"].str.lower().str.contains(q, na=False) |
                  sub["features"].str.lower().str.contains(q, na=False) |
                  sub["description"].str.lower().str.contains(q, na=False)]

    vcol1, vcol2 = st.columns([1, 4])
    with vcol1:
        view_mode = st.radio("View mode", ["🃏 Cards", "📋 Table"], horizontal=True, label_visibility="collapsed")
    with vcol2:
        st.markdown(f"<div style='text-align:right;color:{C_MUTED};font-size:.85rem;padding-top:.3rem'>Showing <b>{len(sub)}</b> of <b>{len(df)}</b> fixtures</div>", unsafe_allow_html=True)

    if sub.empty:
        st.warning("No fixtures matched your selected filters.")
    elif view_mode == "🃏 Cards":
        cols = st.columns(3)
        for i, (_, p) in enumerate(sub.iterrows()):
            with cols[i % 3]:
                # Prepare badges
                cats = [c.strip() for c in (p["categories"] or "").split(",") if c.strip()]
                feats = [f.strip() for f in (p["features"] or "").split(",") if f.strip()][:3]
                w_opts = p.get("wattage_options") or (f"{p['wattage']}W" if p.get("wattage") else "")
                
                badges_html = ""
                for c in cats[:1]: badges_html += f'<span class="badge badge-blue">{c}</span>'
                for f in feats:    badges_html += f'<span class="badge badge-purple">{f}</span>'
                if w_opts:         badges_html += f'<span class="badge badge-green">⚡ {w_opts}</span>'
                if p["has_discount"]: badges_html += f'<span class="badge badge-amber">{p["discount_percentage"]}% OFF</span>'

                sku_code = p["sku"]
                avatar_txt = sku_code[:2] if len(sku_code) >= 2 else "💡"

                st.markdown(f"""
<div class="pcard">
  <div class="pcard-top">
    <div class="avatar">{avatar_txt}</div>
    <div class="pcard-info">
      <div class="pname" title="{p['name']}">{p['name']}</div>
      <div class="psku">SKU: {p['sku']} · {rating_badge(p['rating_score'])}</div>
    </div>
  </div>
  <div style="margin:.25rem 0">{price_badge(p['price_num'])}</div>
  <div style="line-height:1.4">{badges_html}</div>
  <div class="pspec" style="margin-top:.2rem;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">{p['description'] or 'Authoritative Inventaa outdoor and indoor LED fixture.'}</div>
  <div>
    <a class="view-btn" href="{p['url'] or 'https://inventaa.in'}" target="_blank">🌐 View Catalog Page →</a>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        # Table view
        tdf = sub[["sku", "name", "primary_category", "price_num", "rating_score", "wattage_options", "color_options"]].copy()
        tdf.columns = ["SKU", "Product Name", "Category", "Price (INR)", "Rating", "Wattages", "Colors"]
        st.dataframe(tdf, width='stretch', height=550)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SPECIFICATIONS & VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️  Specifications & Variants":
    st.markdown(f"""
    <div class="hero">
      <h1>⚙️ Specifications & Variant Index</h1>
      <p>Granular specification attributes and purchasable variant combinations across all SKUs.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Specification Matrix", "🎨 Purchasable Variants"])

    with tab1:
        scol1, scol2 = st.columns([2, 2])
        with scol1: spec_filter = st.selectbox("Filter Specification Key", ["All"] + (sorted(list(specs_df["spec_key"].unique())) if not specs_df.empty and "spec_key" in specs_df.columns else []))
        with scol2: spec_search = st.text_input("🔍 Search Product or Spec Value...", "", key="s_search")

        sdf = specs_df.copy()
        if spec_filter != "All":
            sdf = sdf[sdf["spec_key"] == spec_filter]
        if spec_search:
            sq = spec_search.lower()
            sdf = sdf[sdf["product_name"].str.lower().str.contains(sq, na=False) |
                      sdf["product_sku"].str.lower().str.contains(sq, na=False) |
                      sdf["spec_value"].str.lower().str.contains(sq, na=False)]

        st.markdown(f"<div style='color:{C_MUTED};font-size:.85rem;margin-bottom:.5rem'>Showing <b>{len(sdf)}</b> specification rows</div>", unsafe_allow_html=True)
        st.dataframe(sdf.rename(columns={"product_sku": "SKU", "product_name": "Product Name", "categories": "Categories", "spec_key": "Attribute", "spec_value": "Value"}), width='stretch', height=500)

    with tab2:
        vcol1, vcol2 = st.columns([2, 2])
        with vcol1: var_search = st.text_input("🔍 Search Variant SKU or Product...", "", key="v_search")
        with vcol2: color_f    = st.selectbox("Color Option Filter", ["All"] + (sorted(list(set(variants_df["color_option"].replace('', pd.NA).dropna()))) if not variants_df.empty and "color_option" in variants_df.columns else []))

        vdf = variants_df.copy()
        if var_search:
            vq = var_search.lower()
            vdf = vdf[vdf["product_name"].str.lower().str.contains(vq, na=False) |
                      vdf["product_sku"].str.lower().str.contains(vq, na=False) |
                      vdf["variant_sku"].str.lower().str.contains(vq, na=False)]
        if color_f != "All":
            vdf = vdf[vdf["color_option"] == color_f]

        st.markdown(f"<div style='color:{C_MUTED};font-size:.85rem;margin-bottom:.5rem'>Showing <b>{len(vdf)}</b> purchasable SKU variants</div>", unsafe_allow_html=True)
        st.dataframe(vdf.rename(columns={"product_sku": "Master SKU", "product_name": "Product Name", "variant_sku": "Variant SKU", "color_option": "Color", "wattage_option": "Wattage", "price": "Price (INR)"}), width='stretch', height=500)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🕸️  Knowledge Graph":
    st.markdown(f"""
    <div class="hero">
      <h1>🕸️ Inventaa Semantic Knowledge Graph</h1>
      <p>Interactive network graph — Product nodes clustered by Category, Use Case, Feature, or Wattage.</p>
    </div>
    """, unsafe_allow_html=True)

    gc1, gc2, gc3, gc4 = st.columns(4)
    with gc1: node_limit = st.slider("Max fixtures", 20, min(200, len(df)), 75)
    with gc2: rel_type   = st.selectbox("Cluster by", ["Category", "Use Case", "Feature", "Wattage"])
    with gc3: cat_f      = st.selectbox("Category filter", ["All"] + all_categories)
    with gc4: min_rat    = st.slider("Min rating (★)", 0.0, 5.0, 0.0, step=0.5)

    sub = df.copy()
    if cat_f != "All":   sub = sub[sub["categories"].str.contains(cat_f, case=False, na=False)]
    if min_rat > 0:      sub = sub[sub["rating_score"].fillna(0) >= min_rat]
    sub = sub.head(node_limit)

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    node_hover = []
    edge_x, edge_y = [], []
    random.seed(42)

    REL_COLORS = {"Category": C_BLUE, "Use Case": C_PURPLE, "Feature": C_GREEN, "Wattage": C_AMBER}
    gcolor = REL_COLORS[rel_type]

    if rel_type == "Category":
        groups = sub["primary_category"].dropna().unique()
    elif rel_type == "Use Case":
        groups = sub["use_cases"].str.split(",").explode().str.strip().replace('', pd.NA).dropna().unique()
    elif rel_type == "Feature":
        groups = sub["features"].str.split(",").explode().str.strip().replace('', pd.NA).dropna().unique()
        groups = [g for g in groups[:12]] # limit top features for graph clarity
    else:
        groups = sub["wattage"].replace(0, pd.NA).dropna().astype(str).unique()
        groups = [f"{g}W" for g in groups]

    group_pos: dict = {}
    r1 = 4.5
    for idx, g in enumerate(groups):
        a = 2 * math.pi * idx / max(len(groups), 1)
        gx, gy = r1 * math.cos(a), r1 * math.sin(a)
        group_pos[g] = (gx, gy)
        node_x.append(gx); node_y.append(gy)
        node_text.append(f"<b>{g}</b>")
        node_hover.append(f"<b>Hub: {g}</b><br>Cluster: {rel_type}")
        node_color.append(gcolor); node_size.append(22)

    r2 = 1.8
    for _, row in sub.iterrows():
        if rel_type == "Category":
            matched = [row["primary_category"]] if row["primary_category"] in group_pos else []
        elif rel_type == "Use Case":
            matched = [u.strip() for u in (row["use_cases"] or "").split(",") if u.strip() in group_pos]
        elif rel_type == "Feature":
            matched = [f.strip() for f in (row["features"] or "").split(",") if f.strip() in group_pos]
        else:
            w_str = f"{row['wattage']}W" if row.get("wattage") else ""
            matched = [w_str] if w_str in group_pos else []

        if not matched: continue

        gx, gy = group_pos[matched[0]]
        a  = random.uniform(0, 2 * math.pi)
        dx = gx + r2 * math.cos(a) * random.uniform(0.3, 1.0)
        dy = gy + r2 * math.sin(a) * random.uniform(0.3, 1.0)
        
        rat = row.get("rating_score")
        sz  = 10 + (int(rat * 2) if pd.notna(rat) and rat > 0 else 0)
        dc  = cat_color(row["primary_category"])

        node_x.append(dx); node_y.append(dy)
        hover = (f"<b>{row['name']}</b><br>SKU: {row['sku']}<br>"
                 f"🏷️ Category: {row['primary_category']}<br>"
                 f"⚡ Price: Rs. {row['price_num']:,}<br>"
                 + (f"★ {rat:.1f} rating" if pd.notna(rat) and rat > 0 else "Unrated"))
        node_text.append(""); node_hover.append(hover)
        node_color.append(dc); node_size.append(sz)

        for m in matched:
            mgx, mgy = group_pos[m]
            edge_x += [mgx, dx, None]; edge_y += [mgy, dy, None]

    fig_g = go.Figure(data=[
        go.Scatter(x=edge_x, y=edge_y, mode="lines",
                   line=dict(width=1, color=C_BORDER2), hoverinfo="none"),
        go.Scatter(x=node_x, y=node_y, mode="markers+text",
                   marker=dict(size=node_size, color=node_color,
                               line=dict(width=1.5, color="#fff")),
                   text=node_text, textposition="top center",
                   hovertext=node_hover, hoverinfo="text")
    ])
    fig_g.update_layout(**CHART_LAYOUT, height=620, showlegend=False)
    fig_g.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig_g.update_yaxes(showgrid=False, zeroline=False, visible=False)
    st.plotly_chart(fig_g, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — CATEGORY & PRODUCT EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍  Category & Product Explorer":
    st.markdown(f"""
    <div class="hero">
      <h1>🔍 Category & Fixture Deep-Dive</h1>
      <p>Granular analytical breakdown and specification inspection for a selected architectural lighting category.</p>
    </div>
    """, unsafe_allow_html=True)

    chosen_cat = st.selectbox("Choose a Lighting Category", all_categories if all_categories else ["General Purpose Lights"])
    cat_prods  = df[df["categories"].str.contains(chosen_cat, case=False, na=False)]

    if cat_prods.empty:
        st.warning(f"No fixtures found under {chosen_cat}.")
    else:
        # ── KPI Row ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fixtures in Category", len(cat_prods))
        c2.metric("Avg Sale Price",       f"Rs. {int(cat_prods['price_num'].replace(0, pd.NA).dropna().mean() or 0):,}")
        c3.metric("Price Range",          f"Rs. {cat_prods['price_num'].min():,} – {cat_prods['price_num'].max():,}")
        c4.metric("Avg Star Rating",      f"★ {cat_prods['rating_score'].replace(0, pd.NA).dropna().mean() or 0.0:.2f}")

        st.markdown('<div class="sh">Price vs. Star Rating Analytics</div>', unsafe_allow_html=True)

        sc_df = cat_prods[cat_prods["price_num"] > 0]
        if not sc_df.empty:
            fig_sc = px.scatter(sc_df, x="price_num", y="rating_score",
                                size="review_count", color="primary_category",
                                hover_name="name", hover_data=["sku", "wattage_options"],
                                color_discrete_sequence=[cat_color(chosen_cat)],
                                labels={"price_num": "Price (INR)", "rating_score": "Star Rating"})
            fig_sc.update_layout(**CHART_LAYOUT, height=350)
            st.plotly_chart(fig_sc, width='stretch')

        st.markdown('<div class="sh">Fixtures in this Collection</div>', unsafe_allow_html=True)

        for _, pr in cat_prods.iterrows():
            with st.expander(f"💡 {pr['name']} — Rs. {pr['price_num']:,} ({pr['sku']})"):
                ecol1, ecol2 = st.columns([2, 1])
                with ecol1:
                    st.markdown(f"**Description:** {pr['description'] or 'Authoritative Inventaa lighting fixture.'}")
                    st.markdown(f"**Categories:** {pr['categories']}")
                    st.markdown(f"**Use Cases:** {pr['use_cases'] or 'General indoor/outdoor lighting'}")
                    st.markdown(f"**Features:** {pr['features'] or 'Standard LED features'}")
                with ecol2:
                    st.markdown(f"**Wattages:** {pr['wattage_options'] or str(pr['wattage']) + 'W' if pr.get('wattage') else 'Standard'}")
                    st.markdown(f"**Color Options:** {pr['color_options'] or 'Cool White, Warm White'}")
                    st.markdown(f"**Rating:** ★ {pr['rating_score']} ({pr['review_count']} reviews)")
                    st.markdown(f"[🌐 Open Product Page]({pr['url'] or 'https://inventaa.in'})")
