"""
End-to-End Evaluation of Two-Stage Search & Re-Ranking Cascade
Evaluates on Amazon ESCI Test Split with standard IR metrics:
  - Recall@K (Binary exact match coverage across candidate retrieval)
  - Graded NDCG@10 (Exact=1.0, Substitute=0.1, Complement=0.01, Irrelevant=0.0)
  - Per-Stage Latency Profiling (P50, P95)
"""

import time
import argparse
from typing import Dict, List, Set, Any
import numpy as np
import polars as pl
from cascade_engine import CascadeSearchPipeline, ESCI_GAINS


def compute_dcg(relevances: List[float], k: int = 10) -> float:
    """Compute Discounted Cumulative Gain at rank K."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    gains = np.array(relevances)
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return float(np.sum(gains / discounts))


def compute_ndcg(ranked_pids: List[str], ground_truth: Dict[str, float], k: int = 10) -> float:
    """Compute Graded Normalized Discounted Cumulative Gain at rank K."""
    actual_gains = [ground_truth.get(pid, 0.0) for pid in ranked_pids[:k]]
    actual_dcg = compute_dcg(actual_gains, k=k)

    # Ideal DCG
    ideal_gains = sorted(ground_truth.values(), reverse=True)
    ideal_dcg = compute_dcg(ideal_gains, k=k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_recall(ranked_pids: List[str], exact_pids: Set[str], k: int) -> float:
    """Compute Recall@K against known Exact positive items."""
    if not exact_pids:
        return 0.0
    hits = sum(1 for pid in ranked_pids[:k] if pid in exact_pids)
    return hits / len(exact_pids)


def run_cascade_evaluation(num_queries: int = 100, candidate_depth: int = 50):
    print("=" * 70)
    print("RUNNING END-TO-END SEARCH CASCADE EVALUATION ON ESCI TEST QUERIES")
    print("=" * 70)

    # 1. Load ESCI US Task 1 test queries & judgments
    print("Loading test judgments...")
    examples_df = pl.read_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
        columns=["query_id", "query", "product_id", "product_locale", "esci_label", "small_version", "split"]
    ).filter(
        (pl.col("product_locale") == "us") &
        (pl.col("small_version") == 1) &
        (pl.col("split") == "test")
    )

    # Filter to queries that have at least one Exact positive
    queries_with_exact = (
        examples_df.filter(pl.col("esci_label").is_in(["E", "Exact"]))
        .select("query_id")
        .unique()["query_id"]
        .to_list()
    )

    test_queries = (
        examples_df.filter(pl.col("query_id").is_in(queries_with_exact))
        .select(["query_id", "query"])
        .unique()
        .head(num_queries)
    )

    eval_query_ids = set(test_queries["query_id"].to_list())
    eval_judgments = examples_df.filter(pl.col("query_id").is_in(list(eval_query_ids)))
    eval_product_ids = set(eval_judgments["product_id"].unique().to_list())

    print(f"Selected {len(eval_query_ids)} test queries with {len(eval_product_ids)} judged products.")

    # 2. Load Product metadata for candidate pool
    print("Loading product metadata for evaluation corpus...")
    products_df = pl.scan_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    ).filter(
        (pl.col("product_locale") == "us") &
        (pl.col("product_id").is_in(list(eval_product_ids)))
    ).collect()

    print(f"Product corpus loaded: {products_df.height} products.")

    # 3. Initialize Cascade Engine
    pipeline = CascadeSearchPipeline(
        products_df,
        candidate_depth=candidate_depth,
        final_top_k=10
    )

    # 4. Group ground truth by query
    # qid -> {pid: gain}
    q_relevance: Dict[int, Dict[str, float]] = {}
    q_exact: Dict[int, Set[str]] = {}
    for row in eval_judgments.to_dicts():
        qid = row["query_id"]
        pid = row["product_id"]
        label = row["esci_label"]
        gain = ESCI_GAINS.get(label, 0.0)

        if qid not in q_relevance:
            q_relevance[qid] = {}
            q_exact[qid] = set()

        q_relevance[qid][pid] = gain
        if label in ("E", "Exact"):
            q_exact[qid].add(pid)

    # 5. Evaluate Queries
    metrics = {
        "bm25_rec10": [], "bm25_rec25": [], "bm25_rec50": [], "bm25_ndcg10": [],
        "dense_rec10": [], "dense_rec25": [], "dense_rec50": [], "dense_ndcg10": [],
        "rrf_rec10": [], "rrf_rec25": [], "rrf_rec50": [], "rrf_ndcg10": [],
        "cascade_ndcg10": [],
        "lat_bm25": [], "lat_dense": [], "lat_fusion": [], "lat_rerank": [], "lat_total": []
    }

    print(f"\nExecuting search & evaluation across {test_queries.height} queries...")
    for i, row in enumerate(test_queries.to_dicts(), start=1):
        qid = row["query_id"]
        query = row["query"]
        exact_pids = q_exact.get(qid, set())
        gt_gains = q_relevance.get(qid, {})

        res = pipeline.search(query)

        bm25_pids = [p for p, _ in res["bm25_top"]]
        dense_pids = [p for p, _ in res["dense_top"]]
        rrf_pids = [p for p, _ in res["hybrid_top"]]
        rerank_pids = [p for p, _ in res["reranked_top"]]

        # Latencies
        t = res["timings"]
        metrics["lat_bm25"].append(t["bm25_ms"])
        metrics["lat_dense"].append(t["dense_ms"])
        metrics["lat_fusion"].append(t["fusion_ms"])
        metrics["lat_rerank"].append(t["reranker_ms"])
        metrics["lat_total"].append(t["total_ms"])

        # Recall
        metrics["bm25_rec10"].append(compute_recall(bm25_pids, exact_pids, 10))
        metrics["bm25_rec25"].append(compute_recall(bm25_pids, exact_pids, 25))
        metrics["bm25_rec50"].append(compute_recall(bm25_pids, exact_pids, 50))

        metrics["dense_rec10"].append(compute_recall(dense_pids, exact_pids, 10))
        metrics["dense_rec25"].append(compute_recall(dense_pids, exact_pids, 25))
        metrics["dense_rec50"].append(compute_recall(dense_pids, exact_pids, 50))

        metrics["rrf_rec10"].append(compute_recall(rrf_pids, exact_pids, 10))
        metrics["rrf_rec25"].append(compute_recall(rrf_pids, exact_pids, 25))
        metrics["rrf_rec50"].append(compute_recall(rrf_pids, exact_pids, 50))

        # NDCG@10
        metrics["bm25_ndcg10"].append(compute_ndcg(bm25_pids, gt_gains, 10))
        metrics["dense_ndcg10"].append(compute_ndcg(dense_pids, gt_gains, 10))
        metrics["rrf_ndcg10"].append(compute_ndcg(rrf_pids, gt_gains, 10))
        metrics["cascade_ndcg10"].append(compute_ndcg(rerank_pids, gt_gains, 10))

    # Print Results Summary
    print("\n" + "=" * 70)
    print("CANDIDATE RETRIEVAL EVALUATION (STAGE 1)")
    print("=" * 70)
    print(f"{'System':<20} | {'Recall@10':<12} | {'Recall@25':<12} | {'Recall@50':<12}")
    print("-" * 70)
    print(f"{'BM25 (Tuned)':<20} | {np.mean(metrics['bm25_rec10']):.4f}       | {np.mean(metrics['bm25_rec25']):.4f}       | {np.mean(metrics['bm25_rec50']):.4f}")
    print(f"{'Dense (Bi-Encoder)':<20} | {np.mean(metrics['dense_rec10']):.4f}       | {np.mean(metrics['dense_rec25']):.4f}       | {np.mean(metrics['dense_rec50']):.4f}")
    print(f"{'Hybrid RRF (Fused)':<20} | {np.mean(metrics['rrf_rec10']):.4f}       | {np.mean(metrics['rrf_rec25']):.4f}       | {np.mean(metrics['rrf_rec50']):.4f}")

    print("\n" + "=" * 70)
    print("RANKING & RE-RANKING QUALITY (STAGE 2)")
    print("=" * 70)
    bm25_ndcg = np.mean(metrics['bm25_ndcg10'])
    dense_ndcg = np.mean(metrics['dense_ndcg10'])
    rrf_ndcg = np.mean(metrics['rrf_ndcg10'])
    cascade_ndcg = np.mean(metrics['cascade_ndcg10'])

    print(f"{'Stage / Method':<30} | {'NDCG@10':<10} | {'Gain vs BM25':<12}")
    print("-" * 70)
    print(f"{'BM25 Baseline':<30} | {bm25_ndcg:.4f}     | {'Baseline':<12}")
    print(f"{'Dense Bi-Encoder':<30} | {dense_ndcg:.4f}     | {dense_ndcg - bm25_ndcg:+.4f} ({((dense_ndcg-bm25_ndcg)/bm25_ndcg)*100:+.1f}%)")
    print(f"{'Hybrid RRF':<30} | {rrf_ndcg:.4f}     | {rrf_ndcg - bm25_ndcg:+.4f} ({((rrf_ndcg-bm25_ndcg)/bm25_ndcg)*100:+.1f}%)")
    print(f"{'Full Cascade (Re-Ranked)':<30} | {cascade_ndcg:.4f}     | {cascade_ndcg - bm25_ndcg:+.4f} ({((cascade_ndcg-bm25_ndcg)/bm25_ndcg)*100:+.1f}%)")

    print("\n" + "=" * 70)
    print("LATENCY BENCHMARK PROFILE (ms per query)")
    print("=" * 70)
    print(f"{'Stage':<25} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10}")
    print("-" * 70)
    for name, key in [("BM25 Lexical", "lat_bm25"), ("Dense Vector", "lat_dense"),
                      ("Hybrid RRF", "lat_fusion"), ("Cross-Encoder Re-Rank", "lat_rerank"),
                      ("Total End-to-End", "lat_total")]:
        vals = metrics[key]
        print(f"{name:<25} | {np.mean(vals):.2f}       | {np.percentile(vals, 50):.2f}       | {np.percentile(vals, 95):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=100, help="Number of test queries to evaluate")
    parser.add_argument("--depth", type=int, default=50, help="Candidate retrieval depth")
    args = parser.parse_args()

    run_cascade_evaluation(num_queries=args.queries, candidate_depth=args.depth)
