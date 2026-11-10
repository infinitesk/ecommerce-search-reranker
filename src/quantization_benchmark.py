"""
Cross-Encoder Model Optimization & Dynamic INT8 Quantization Benchmark
Quantizes the second-stage re-ranker to INT8 and benchmarks:
  1. Model Memory / File Size (FP32 vs INT8)
  2. Scoring Latency per Candidate Batch (ms)
  3. Ranking Quality (NDCG@10 Retention)
"""

import os
import time
from typing import Dict, List
import numpy as np
import polars as pl
import torch
from sentence_transformers import CrossEncoder
from evaluate_cascade import compute_ndcg
from cascade_engine import ESCI_GAINS, build_product_doc_text


def benchmark_quantization(num_test_queries: int = 30):
    print("=" * 75)
    print("CROSS-ENCODER INT8 DYNAMIC QUANTIZATION & LATENCY BENCHMARK")
    print("=" * 75)

    device = "cpu"
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    print("Loading Baseline FP32 Cross-Encoder...")
    fp32_model = CrossEncoder(model_name, device=device)

    # 1. Apply Dynamic INT8 Quantization to Linear layers
    print("Applying PyTorch Dynamic INT8 Quantization (qnnpack engine)...")
    torch.backends.quantized.engine = 'qnnpack'
    int8_model = CrossEncoder(model_name, device=device)
    int8_model.model = torch.ao.quantization.quantize_dynamic(
        int8_model.model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # Measure model file size using state_dict bytes
    import io
    def get_model_size_mb(model):
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.tell() / (1024 * 1024)

    fp32_size = get_model_size_mb(fp32_model.model)
    int8_size = get_model_size_mb(int8_model.model)
    print(f"\nModel Size: FP32 = {fp32_size:.1f} MB | INT8 = {int8_size:.1f} MB (Compression: {fp32_size/max(int8_size,1e-6):.1f}x)")

    # 2. Load ESCI Test judgments
    print("\nLoading ESCI test queries for latency and NDCG verification...")
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

    products_df = pl.scan_parquet(
        "shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    ).filter(
        (pl.col("product_locale") == "us") &
        (pl.col("product_id").is_in(list(eval_pids)))
    ).collect()

    product_text_map = {
        row["product_id"]: build_product_doc_text(
            row.get("product_title"),
            row.get("product_brand"),
            row.get("product_color"),
            row.get("product_bullet_point"),
            row.get("product_description")
        )
        for row in products_df.to_dicts()
    }

    q_relevance: Dict[int, Dict[str, float]] = {}
    q_products: Dict[int, List[str]] = {}
    for row in eval_judgments.to_dicts():
        qid = row["query_id"]
        pid = row["product_id"]
        gain = ESCI_GAINS.get(row["esci_label"], 0.0)
        if qid not in q_relevance:
            q_relevance[qid] = {}
            q_products[qid] = []
        q_relevance[qid][pid] = gain
        q_products[qid].append(pid)

    # 3. Latency & Accuracy Benchmark
    fp32_latencies = []
    int8_latencies = []
    fp32_ndcgs = []
    int8_ndcgs = []

    print(f"Scoring {test_queries.height} queries with FP32 and INT8 models...")

    for row in test_queries.to_dicts():
        qid = row["query_id"]
        query = row["query"]
        pids = q_products[qid]
        pairs = [(query, product_text_map.get(pid, "")) for pid in pids]

        # Warmup / FP32 scoring
        t0 = time.perf_counter()
        fp32_scores = fp32_model.predict(pairs, show_progress_bar=False)
        fp32_lat = (time.perf_counter() - t0) * 1000
        fp32_latencies.append(fp32_lat)

        # INT8 scoring
        t1 = time.perf_counter()
        int8_scores = int8_model.predict(pairs, show_progress_bar=False)
        int8_lat = (time.perf_counter() - t1) * 1000
        int8_latencies.append(int8_lat)

        # NDCG@10
        fp32_ranked = [p for p, _ in sorted(zip(pids, fp32_scores), key=lambda x: x[1], reverse=True)]
        int8_ranked = [p for p, _ in sorted(zip(pids, int8_scores), key=lambda x: x[1], reverse=True)]

        fp32_ndcgs.append(compute_ndcg(fp32_ranked, q_relevance[qid], k=10))
        int8_ndcgs.append(compute_ndcg(int8_ranked, q_relevance[qid], k=10))

    # Print Comparison Table
    print("\n" + "=" * 75)
    print("QUANTIZATION BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<30} | {'FP32 Baseline':<16} | {'INT8 Quantized':<16} | {'Change':<12}")
    print("-" * 75)
    print(f"{'Model Size':<30} | {fp32_size:6.1f} MB         | {int8_size:6.1f} MB         | -{(1 - int8_size/fp32_size)*100:.1f}%")
    print(f"{'Mean Latency per Query':<30} | {np.mean(fp32_latencies):6.2f} ms        | {np.mean(int8_latencies):6.2f} ms        | -{(1 - np.mean(int8_latencies)/np.mean(fp32_latencies))*100:.1f}% ({np.mean(fp32_latencies)/np.mean(int8_latencies):.2f}x speedup)")
    print(f"{'P50 Latency (Median)':<30} | {np.percentile(fp32_latencies, 50):6.2f} ms        | {np.percentile(int8_latencies, 50):6.2f} ms        | -{(1 - np.percentile(int8_latencies, 50)/np.percentile(fp32_latencies, 50))*100:.1f}%")
    print(f"{'P95 Latency':<30} | {np.percentile(fp32_latencies, 95):6.2f} ms        | {np.percentile(int8_latencies, 95):6.2f} ms        | -{(1 - np.percentile(int8_latencies, 95)/np.percentile(fp32_latencies, 95))*100:.1f}%")
    print(f"{'NDCG@10 Quality':<30} | {np.mean(fp32_ndcgs):6.4f}           | {np.mean(int8_ndcgs):6.4f}           | {(np.mean(int8_ndcgs) - np.mean(fp32_ndcgs)):+.4f} ({(np.mean(int8_ndcgs)/np.mean(fp32_ndcgs) - 1)*100:+.2f}%)")
    print("=" * 75)


if __name__ == "__main__":
    benchmark_quantization(num_test_queries=30)
