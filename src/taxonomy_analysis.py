"""
ESCI Query Taxonomy & Failure Mode Analysis
Investigates model behavior across distinct challenge query distributions:
  1. negations: queries containing negative constraints (e.g., 'without rims', 'no sugar')
  2. parse_pattern: queries specifying dimensions / technical attributes (e.g., '15 inch', '0.12 bbs')
  3. behavioral: intent / price oriented queries (e.g., '$50 gifts for men')
  4. nlqec: conversational natural language queries
  5. other: head and general e-commerce queries
"""

import time
from typing import Dict, List, Set
import numpy as np
import polars as pl
from cascade_engine import CascadeSearchPipeline, ESCI_GAINS
from evaluate_cascade import compute_ndcg, compute_recall


def run_taxonomy_analysis(queries_per_category: int = 25):
    print("=" * 75)
    print("RUNNING ESCI QUERY TAXONOMY & FAILURE MODE ANALYSIS")
    print("=" * 75)

    # 1. Load sources
    sources_df = pl.read_csv("shopping_queries_dataset/shopping_queries_dataset_sources.csv")

    # 2. Load Task 1 US examples
    examples_df = pl.read_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
        columns=["query_id", "query", "product_id", "product_locale", "esci_label", "small_version", "split"]
    ).filter(
        (pl.col("product_locale") == "us") &
        (pl.col("small_version") == 1)
    )

    # Join with sources
    annotated_df = examples_df.join(sources_df, on="query_id", how="inner")

    # Group ground truth by query_id
    q_info = {}
    for row in annotated_df.to_dicts():
        qid = row["query_id"]
        if qid not in q_info:
            q_info[qid] = {
                "query": row["query"],
                "source": row["source"],
                "split": row["split"],
                "judgments": {},
                "exact": set()
            }
        q_info[qid]["judgments"][row["product_id"]] = ESCI_GAINS.get(row["esci_label"], 0.0)
        if row["esci_label"] in ("E", "Exact"):
            q_info[qid]["exact"].add(row["product_id"])

    # Sample queries per taxonomy category with at least 1 exact match
    categories = ["negations", "parse_pattern", "behavioral", "nlqec", "other"]
    sampled_queries = []
    for cat in categories:
        candidates = [
            (qid, data["query"], cat)
            for qid, data in q_info.items()
            if data["source"] == cat and len(data["exact"]) > 0
        ]
        # Take up to queries_per_category
        sampled_queries.extend(candidates[:queries_per_category])

    print(f"Sampled {len(sampled_queries)} total queries across {len(categories)} taxonomies.")

    # Collect product ids needed
    eval_qids = {qid for qid, _, _ in sampled_queries}
    needed_pids = set()
    for qid in eval_qids:
        needed_pids.update(q_info[qid]["judgments"].keys())

    # Load product catalog subset
    products_df = pl.scan_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    ).filter(
        (pl.col("product_locale") == "us") &
        (pl.col("product_id").is_in(list(needed_pids)))
    ).collect()

    print(f"Corpus size: {products_df.height} products.")

    pipeline = CascadeSearchPipeline(products_df, candidate_depth=40, final_top_k=10)

    # Run evaluation stratified by category
    results_by_cat = {cat: {"bm25": [], "dense": [], "cascade": []} for cat in categories}

    print("\nRunning taxonomy queries through the cascade...")
    for qid, query, cat in sampled_queries:
        data = q_info[qid]
        res = pipeline.search(query)

        bm25_pids = [p for p, _ in res["bm25_top"]]
        dense_pids = [p for p, _ in res["dense_top"]]
        cascade_pids = [p for p, _ in res["reranked_top"]]

        bm25_ndcg = compute_ndcg(bm25_pids, data["judgments"], k=10)
        dense_ndcg = compute_ndcg(dense_pids, data["judgments"], k=10)
        cascade_ndcg = compute_ndcg(cascade_pids, data["judgments"], k=10)

        results_by_cat[cat]["bm25"].append(bm25_ndcg)
        results_by_cat[cat]["dense"].append(dense_ndcg)
        results_by_cat[cat]["cascade"].append(cascade_ndcg)

    # Print Taxonomy Breakdown Table
    print("\n" + "=" * 80)
    print("TAXONOMY PERFORMANCE BREAKDOWN (NDCG@10)")
    print("=" * 80)
    print(f"{'Category':<15} | {'N Queries':<10} | {'BM25':<10} | {'Dense':<10} | {'Cascade (CE)':<14} | {'Gain vs BM25':<12}")
    print("-" * 80)

    for cat in categories:
        scores = results_by_cat[cat]
        n_q = len(scores["bm25"])
        if n_q == 0:
            continue
        m_bm25 = np.mean(scores["bm25"])
        m_dense = np.mean(scores["dense"])
        m_cascade = np.mean(scores["cascade"])
        gain = m_cascade - m_bm25
        print(f"{cat:<15} | {n_q:<10} | {m_bm25:.4f}     | {m_dense:.4f}     | {m_cascade:.4f}         | {gain:+.4f} ({gain/max(m_bm25, 1e-6)*100:+.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_taxonomy_analysis(queries_per_category=20)
