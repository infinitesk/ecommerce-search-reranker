"""
Benchmark: Amazon Science Official Baseline (MiniLM-L-12-v2) vs. L-6 Base vs. L-6 Full Fine-Tuned
Evaluates NDCG@10, Parameter Count, and Inference Latency on held-out ESCI test queries.
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


def count_parameters(model):
    return sum(p.numel() for p in model.model.parameters())


def benchmark_models(num_test_queries: int = 100):
    print("=" * 85)
    print("BENCHMARK: AMAZON SCIENCE L-12 VS. L-6 BASE VS. FULL FINE-TUNED L-6")
    print("=" * 85)

    models_config = [
        {
            "name": "Base MiniLM-L-6-v2",
            "path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "layers": 6,
            "description": "General domain MS-MARCO 6-layer base"
        },
        {
            "name": "Base MiniLM-L-12-v2 (Amazon Baseline)",
            "path": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "layers": 12,
            "description": "Amazon Science official benchmark base"
        },
        {
            "name": "ESCI Fine-Tuned MiniLM-L-6-v2 (Ours)",
            "path": "models/esci_cross_encoder_full",
            "layers": 6,
            "description": "Fine-tuned on 419k ESCI product pairs"
        }
    ]

    loaded_models = []
    for cfg in models_config:
        print(f"Loading {cfg['name']}...")
        model = CrossEncoder(cfg["path"], device="cpu")
        param_count = count_parameters(model)
        loaded_models.append({
            **cfg,
            "model": model,
            "params": param_count
        })

    # Load ESCI Test Judgments
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

    # Initialize results tracking
    results = {
        m["name"]: {"ndcgs": [], "latencies": [], "params": m["params"], "layers": m["layers"]}
        for m in loaded_models
    }

    print("\nScoring test queries across all models...")
    for qid, query in zip(test_queries["query_id"].to_list(), test_queries["query"].to_list()):
        pids = q_candidates[qid]
        pairs = [(query, prod_title_map.get(pid, "")) for pid in pids]

        for m in loaded_models:
            t0 = time.perf_counter()
            scores = m["model"].predict(pairs, show_progress_bar=False)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            ranked = [p for p, _ in sorted(zip(pids, scores), key=lambda x: x[1], reverse=True)]
            ndcg = compute_ndcg(ranked, q_relevance[qid], k=10)

            results[m["name"]]["ndcgs"].append(ndcg)
            results[m["name"]]["latencies"].append(latency_ms)

    print("\n" + "=" * 95)
    print("BENCHMARK SUMMARY RESULTS (NDCG@10 & LATENCY ON CPU)")
    print("=" * 95)
    print(f"{'Model':<38} | {'Params':<9} | {'Layers':<6} | {'NDCG@10':<9} | {'Latency/Q':<11} | {'vs Base L-6'}")
    print("-" * 95)

    base_ndcg = np.mean(results["Base MiniLM-L-6-v2"]["ndcgs"])
    for name, data in results.items():
        m_ndcg = np.mean(data["ndcgs"])
        m_lat = np.mean(data["latencies"])
        delta = m_ndcg - base_ndcg
        pct = (delta / base_ndcg) * 100
        delta_str = f"{delta:+.4f} ({pct:+.1f}%)" if name != "Base MiniLM-L-6-v2" else "Baseline"
        params_m = f"{data['params'] / 1e6:.1f}M"
        print(f"{name:<38} | {params_m:<9} | {data['layers']:<6} | {m_ndcg:.4f}   | {m_lat:.1f} ms    | {delta_str}")
    print("=" * 95)


if __name__ == "__main__":
    benchmark_models(num_test_queries=100)
