"""
Direct Before-vs-After Comparison on Held-Out ESCI Test Queries
Benchmarks:
  1. Base Cross-Encoder (Before Fine-Tuning)
  2. Full Dataset Fine-Tuned Cross-Encoder (After 419,653 pairs)
  3. LightGBM LambdaRank (Small vs Large Scale)
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "8"
import time
import numpy as np
import polars as pl
import torch
torch.set_num_threads(8)
from sentence_transformers import CrossEncoder
from evaluate_cascade import compute_ndcg
from cascade_engine import clean_text, ESCI_GAINS


def evaluate_before_after(num_test_queries: int = 100):
    print("=" * 75)
    print("BEFORE VS. AFTER FULL FINE-TUNING EVALUATION ON TEST SET")
    print("=" * 75)

    # 1. Load Models
    print("Loading Base Cross-Encoder (before fine-tuning)...")
    base_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    full_ft_path = "models/esci_cross_encoder_full"
    print(f"Loading Full-Dataset Fine-Tuned Cross-Encoder from {full_ft_path}...")
    ft_model = CrossEncoder(full_ft_path, device="cpu")

    # 2. Load ESCI Test Judgments
    print("\nLoading held-out test queries...")
    examples_df = pl.read_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
        columns=["query_id", "query", "product_id", "product_locale", "esci_label", "small_version", "split"]
    ).filter(
        (pl.col("product_locale") == "us") &
        (pl.col("small_version") == 1) &
        (pl.col("split") == "test")
    )

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
        .head(num_test_queries)
    )

    eval_qids = set(test_queries["query_id"].to_list())
    eval_judgments = examples_df.filter(pl.col("query_id").is_in(list(eval_qids)))
    eval_pids = set(eval_judgments["product_id"].unique().to_list())

    print(f"Selected {len(eval_qids)} test queries with {len(eval_pids)} judged products.")

    products_raw = pl.scan_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    ).filter(
        (pl.col("product_locale") == "us") &
        (pl.col("product_id").is_in(list(eval_pids)))
    ).collect()

    prod_title_map = dict(zip(
        products_raw["product_id"].to_list(),
        [clean_text(t) for t in products_raw["product_title"].to_list()]
    ))

    # Group ground truth by query
    q_relevance = {}
    q_candidates = {}
    for row in eval_judgments.to_dicts():
        qid = row["query_id"]
        pid = row["product_id"]
        if qid not in q_relevance:
            q_relevance[qid] = {}
            q_candidates[qid] = []
        q_relevance[qid][pid] = ESCI_GAINS.get(row["esci_label"], 0.0)
        q_candidates[qid].append(pid)

    base_ndcgs = []
    ft_ndcgs = []

    print("\nScoring test queries side-by-side...")
    for qid, query in zip(test_queries["query_id"].to_list(), test_queries["query"].to_list()):
        pids = q_candidates[qid]
        pairs = [(query, prod_title_map.get(pid, "")) for pid in pids]

        # 1. Base model predictions
        scores_base = base_model.predict(pairs, show_progress_bar=False)
        ranked_base = [p for p, _ in sorted(zip(pids, scores_base), key=lambda x: x[1], reverse=True)]

        # 2. Fine-tuned model predictions
        scores_ft = ft_model.predict(pairs, show_progress_bar=False)
        ranked_ft = [p for p, _ in sorted(zip(pids, scores_ft), key=lambda x: x[1], reverse=True)]

        base_ndcgs.append(compute_ndcg(ranked_base, q_relevance[qid], k=10))
        ft_ndcgs.append(compute_ndcg(ranked_ft, q_relevance[qid], k=10))

    m_base = np.mean(base_ndcgs)
    m_ft = np.mean(ft_ndcgs)
    delta = m_ft - m_base

    print("\n" + "=" * 75)
    print("CROSS-ENCODER: BEFORE VS. AFTER FULL FINE-TUNING")
    print("=" * 75)
    print(f"{'Model State':<35} | {'NDCG@10':<12} | {'Improvement':<16}")
    print("-" * 75)
    print(f"{'Base MS-MARCO MiniLM (Before)':<35} | {m_base:.4f}       | {'Baseline':<16}")
    print(f"{'ESCI Full Fine-Tuned (After 419k)':<35} | {m_ft:.4f}       | {delta:+.4f} ({(delta/m_base)*100:+.2f}%)")
    print("=" * 75)


if __name__ == "__main__":
    evaluate_before_after(num_test_queries=100)
