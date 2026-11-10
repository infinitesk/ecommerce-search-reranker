"""
Interactive Two-Stage Product Search Demo
Provides real-time side-by-side search comparison:
  [BM25] vs [Dense Bi-Encoder] vs [Hybrid RRF] vs [Cross-Encoder Re-Ranker]
with latency profiling and ground-truth ESCI label indicators.
"""

import sys
import time
import argparse
import polars as pl
from cascade_engine import CascadeSearchPipeline


def load_demo_catalog(sample_size: int = 5000):
    print(f"Loading {sample_size} product catalog subset for fast interactive search...")
    # Read product catalog
    products_df = pl.read_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    ).filter(
        pl.col("product_locale") == "us"
    ).head(sample_size)
    return products_df


def run_interactive_demo():
    print("=" * 75)
    print("AMAZON ESCI TWO-STAGE SEARCH & RE-RANKING ENGINE DEMO")
    print("=" * 75)

    products_df = load_demo_catalog(sample_size=3000)
    pipeline = CascadeSearchPipeline(products_df, candidate_depth=30, final_top_k=5)

    # Load example query annotations for reference lookup
    examples_df = pl.read_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
        columns=["query", "product_id", "esci_label"]
    )
    # Map (query.lower(), product_id) -> label
    label_lookup = {
        (row["query"].strip().lower(), row["product_id"]): row["esci_label"]
        for row in examples_df.head(100000).to_dicts()
    }

    sample_queries = [
        "canon 5d mark iv",
        "wireless earbuds waterproof",
        "lawnmower tires without rims",
        "15 inch laptop 16gb ram",
        "organic green tea bags"
    ]

    print("\nPre-configured test queries:")
    for idx, q in enumerate(sample_queries, start=1):
        print(f"  [{idx}] {q}")

    print("\nEnter a number (1-5), type your own query, or 'q' to exit.\n")

    while True:
        try:
            user_input = input("Search query > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() == 'q':
            print("Exiting search demo.")
            break

        if user_input in [str(i) for i in range(1, len(sample_queries) + 1)]:
            query = sample_queries[int(user_input) - 1]
        else:
            query = user_input

        print(f"\nSearching for: '{query}'...")
        res = pipeline.search(query)
        t = res["timings"]
        title_map = res["product_title_map"]
        q_norm = query.strip().lower()

        print("\n" + "-" * 75)
        print(f"LATENCY: BM25: {t['bm25_ms']:.1f}ms | Dense: {t['dense_ms']:.1f}ms | RRF: {t['fusion_ms']:.1f}ms | Re-Ranker: {t['reranker_ms']:.1f}ms | TOTAL: {t['total_ms']:.1f}ms")
        print("-" * 75)

        for stage_name, hits in [
            ("Stage 1a: Lexical BM25 (Tuned)", res["bm25_top"]),
            ("Stage 1b: Dense Bi-Encoder", res["dense_top"]),
            ("Stage 1c: Hybrid RRF Fusion", res["hybrid_top"]),
            ("Stage 2: Cross-Encoder Re-Ranked", res["reranked_top"])
        ]:
            print(f"\n{stage_name}:")
            for rank, (pid, score) in enumerate(hits, start=1):
                title = title_map.get(pid, "Unknown Title")[:65]
                label = label_lookup.get((q_norm, pid), "Unjudged")
                label_badge = f"[{label}]" if label != "Unjudged" else ""
                print(f"  {rank}. {score:6.3f} | {pid} | {title} {label_badge}")
        print("\n" + "=" * 75)


if __name__ == "__main__":
    run_interactive_demo()
