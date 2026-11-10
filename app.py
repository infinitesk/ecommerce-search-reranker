"""
Streamlit Web UI for E-Commerce Product Search & Re-Ranking
Two-Stage Cascade (BM25 + Dense Bi-Encoder + RRF + Cross-Encoder)
Amazon ESCI Benchmark
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import time
import polars as pl
import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Amazon ESCI Semantic Product Search",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-exact { background-color: #28a745; color: white; }
    .badge-substitute { background-color: #ffc107; color: black; }
    .badge-complement { background-color: #17a2b8; color: white; }
    .badge-irrelevant { background-color: #dc3545; color: white; }
    .badge-unjudged { background-color: #6c757d; color: white; }
    .product-card {
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading product catalog and models...")
def get_pipeline():
    from cascade_engine import CascadeSearchPipeline
    products_df = pl.read_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    ).filter(
        pl.col("product_locale") == "us"
    ).head(4000)

    pipeline = CascadeSearchPipeline(
        products_df,
        candidate_depth=30,
        final_top_k=8
    )
    return pipeline, products_df


@st.cache_data
def get_label_lookup():
    examples_df = pl.read_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
        columns=["query", "product_id", "esci_label"]
    )
    lookup = {
        (row["query"].strip().lower(), row["product_id"]): row["esci_label"]
        for row in examples_df.head(150000).to_dicts()
    }
    return lookup


st.title("🛒 Two-Stage E-Commerce Search & Re-Ranking Engine")
st.markdown(
    "**End-to-End Search Cascade on the Amazon ESCI Benchmark** | "
    "Lexical BM25 ($k_1=0.5, b=1.0$) + Dense Bi-Encoder + Reciprocal Rank Fusion + Cross-Encoder"
)

# Sidebar
st.sidebar.header("⚙️ Search Architecture Settings")
candidate_depth = st.sidebar.slider("Stage 1 Retrieval Depth", 10, 50, 30, step=5)
final_k = st.sidebar.slider("Stage 2 Re-Ranked Top-K", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏷️ ESCI Relevance Legend")
st.sidebar.markdown("""
- <span class="metric-badge badge-exact">E</span> **Exact:** Exact match to customer intent
- <span class="metric-badge badge-substitute">S</span> **Substitute:** Functional alternative
- <span class="metric-badge badge-complement">C</span> **Complement:** Accessory / Bought together
- <span class="metric-badge badge-irrelevant">I</span> **Irrelevant:** Does not satisfy query
""", unsafe_allow_html=True)

# Sample query buttons
st.markdown("##### 🔍 Quick Examples:")
cols = st.columns(5)
sample_queries = [
    "canon 5d mark iv",
    "lawnmower tires without rims",
    "wireless earbuds waterproof",
    "15 inch laptop 16gb ram",
    "$50 gifts for men"
]

selected_query = None
for i, q in enumerate(sample_queries):
    if cols[i].button(q, key=f"btn_{i}", use_container_width=True):
        selected_query = q

query_input = st.text_input(
    "Enter product search query:",
    value=selected_query if selected_query else "canon 5d mark iv"
)

if query_input:
    pipeline, products_df = get_pipeline()
    pipeline.candidate_depth = candidate_depth
    pipeline.final_top_k = final_k
    label_lookup = get_label_lookup()

    # Run search
    res = pipeline.search(query_input)
    t = res["timings"]
    q_norm = query_input.strip().lower()

    # Latency Metrics Banner
    st.markdown("### ⚡ Latency Telemetry Breakdown")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("1. Lexical BM25", f"{t['bm25_ms']:.2f} ms")
    m_col2.metric("2. Dense Vector", f"{t['dense_ms']:.2f} ms")
    m_col3.metric("3. RRF Fusion", f"{t['fusion_ms']:.2f} ms")
    m_col4.metric("4. Cross-Encoder", f"{t['reranker_ms']:.2f} ms")
    m_col5.metric("Total End-to-End", f"{t['total_ms']:.2f} ms")

    # Render Side-by-Side Results
    st.markdown("---")
    st.markdown("### 📊 Stage-by-Stage Results Comparison")

    col_bm25, col_dense, col_rrf, col_rerank = st.columns(4)

    def render_hits_column(column, title, hits, badge_color):
        with column:
            st.markdown(f"#### {title}")
            for rank, (pid, score) in enumerate(hits, start=1):
                # Retrieve product metadata
                p_match = products_df.filter(pl.col("product_id") == pid)
                if p_match.height > 0:
                    row = p_match.to_dicts()[0]
                    p_title = row.get("product_title", "Unknown")
                    p_brand = row.get("product_brand") or "Generic"
                else:
                    p_title = res["product_title_map"].get(pid, "Unknown Product")
                    p_brand = "Unknown"

                raw_label = label_lookup.get((q_norm, pid))
                if raw_label in ("E", "Exact"):
                    badge = '<span class="metric-badge badge-exact">Exact</span>'
                elif raw_label in ("S", "Substitute"):
                    badge = '<span class="metric-badge badge-substitute">Substitute</span>'
                elif raw_label in ("C", "Complement"):
                    badge = '<span class="metric-badge badge-complement">Complement</span>'
                elif raw_label in ("I", "Irrelevant"):
                    badge = '<span class="metric-badge badge-irrelevant">Irrelevant</span>'
                else:
                    badge = '<span class="metric-badge badge-unjudged">Unjudged</span>'

                st.markdown(f"""
                <div class="product-card">
                    <div style="font-size: 0.85rem; color: #555;">#{rank} • Score: {score:.3f} • {p_brand}</div>
                    <div style="font-weight: 600; font-size: 0.95rem; margin: 0.3rem 0;">{p_title[:75]}...</div>
                    <div>{badge} <span style="font-size: 0.75rem; color: #888;">ID: {pid}</span></div>
                </div>
                """, unsafe_allow_html=True)

    render_hits_column(col_bm25, "1. Tuned BM25", res["bm25_top"], "#007bff")
    render_hits_column(col_dense, "2. Dense Bi-Encoder", res["dense_top"], "#6610f2")
    render_hits_column(col_rrf, "3. Hybrid RRF", res["hybrid_top"], "#fd7e14")
    render_hits_column(col_rerank, "4. Re-Ranked (CE)", res["reranked_top"], "#28a745")
