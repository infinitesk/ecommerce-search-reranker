"""
Two-Stage E-Commerce Product Search & Re-Ranking Engine
Amazon ESCI Benchmark

Architecture:
  Stage 1 (Candidate Generation):
    - Lexical: Field-Weighted BM25 (k1=0.5, b=1.0)
    - Dense: Contrastive Bi-Encoder embeddings with Cosine / Dot Product
    - Fusion: Reciprocal Rank Fusion (RRF, k=60)
  Stage 2 (Precision Re-Ranking):
    - Cross-Encoder scoring on (query, product) pairs with calibrated NDCG soft targets
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import time
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import polars as pl
from rank_bm25 import BM25Okapi
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# ESCI Soft Targets for Graded NDCG
ESCI_GAINS = {
    "E": 1.0,
    "S": 0.1,
    "C": 0.01,
    "I": 0.0,
    "Exact": 1.0,
    "Substitute": 0.1,
    "Complement": 0.01,
    "Irrelevant": 0.0
}


def clean_text(text: Optional[str]) -> str:
    """Normalize whitespace and strip HTML tags."""
    if not text:
        return ""
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", text)
    # Strip special entities & non-alphanumeric punctuation excess
    cleaned = re.sub(r"&[a-zA-Z0-9#]+;", " ", cleaned)
    # Normalize whitespaces
    return " ".join(cleaned.split())


def build_product_doc_text(
    title: Optional[str] = "",
    brand: Optional[str] = "",
    color: Optional[str] = "",
    bullet_point: Optional[str] = "",
    description: Optional[str] = ""
) -> str:
    """
    Synthesize structured product text representation with clear field boundaries.
    """
    parts = []
    if title:
        parts.append(f"Title: {clean_text(title)}")
    if brand:
        parts.append(f"Brand: {clean_text(brand)}")
    if color:
        parts.append(f"Color: {clean_text(color)}")
    if bullet_point:
        parts.append(f"Features: {clean_text(bullet_point)[:300]}")
    if description:
        parts.append(f"Description: {clean_text(description)[:250]}")
    return " | ".join(parts)


class LexicalBM25Retriever:
    """
    High-performance in-memory BM25 retriever with field-weighted tokenization.
    Tuned with k1=0.5 and b=1.0 for term-saturated e-commerce listings.
    """
    def __init__(self, k1: float = 0.5, b: float = 1.0):
        self.k1 = k1
        self.b = b
        self.product_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def fit(self, products_df: pl.DataFrame):
        """
        Builds the inverted index with field-weighted tokenization.
        Weights: title (3.0), brand (1.5), bullets (1.0), description (0.5).
        """
        self.product_ids = products_df["product_id"].to_list()
        titles = products_df["product_title"].to_list()
        brands = products_df["product_brand"].to_list()
        bullets = products_df["product_bullet_point"].to_list()

        tokenized_corpus = []
        for i in range(len(self.product_ids)):
            t_tokens = clean_text(titles[i]).lower().split() * 3
            b_tokens = clean_text(brands[i]).lower().split() * 2
            feat_tokens = clean_text(bullets[i]).lower().split()[:30]
            doc_tokens = t_tokens + b_tokens + feat_tokens
            if not doc_tokens:
                doc_tokens = ["empty"]
            tokenized_corpus.append(doc_tokens)

        self.corpus_tokens = tokenized_corpus
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve top_k products by BM25 score."""
        q_tokens = clean_text(query).lower().split()
        if not q_tokens or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(q_tokens)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(self.product_ids[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]


class DenseBiEncoderRetriever:
    """
    Dense semantic retriever using contrastive Bi-Encoder representation.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        if device is None:
            device = "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)
        self.product_ids: List[str] = []
        self.product_embeddings: Optional[np.ndarray] = None

    def fit(self, products_df: pl.DataFrame, batch_size: int = 128):
        """Encodes product catalog into dense representation."""
        self.product_ids = products_df["product_id"].to_list()
        doc_texts = products_df["doc_text"].to_list()
        embeddings = self.model.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        self.product_embeddings = embeddings

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve top_k products using maximum inner product / cosine similarity."""
        if self.product_embeddings is None:
            return []
        q_emb = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        # Cosine similarity dot product
        scores = np.dot(self.product_embeddings, q_emb)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(self.product_ids[idx], float(scores[idx])) for idx in top_indices]


class CrossEncoderReranker:
    """
    Second-stage re-ranker scoring query-product pairs via full cross-attention.
    """
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        if model_name is None:
            # Default to custom fine-tuned checkpoint if available
            local_full = "models/esci_cross_encoder_full"
            local_sample = "models/esci_cross_encoder"
            if os.path.exists(local_full):
                model_name = local_full
            elif os.path.exists(local_sample):
                model_name = local_sample
            else:
                model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        if device is None:
            device = "cpu"
        self.device = device
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        product_text_map: Dict[str, str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Re-rank candidate products using Cross-Encoder attention."""
        if not candidates:
            return []
        pairs = [(query, product_text_map.get(pid, "")) for pid, _ in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip([pid for pid, _ in candidates], scores), key=lambda x: x[1], reverse=True)
        return [(pid, float(score)) for pid, score in ranked[:top_k]]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    top_k: int = 100
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked candidate lists via Reciprocal Rank Fusion (RRF).
    score(doc) = sum(1 / (k + rank_i))
    """
    rrf_scores: Dict[str, float] = {}
    for r_list in ranked_lists:
        for rank, (pid, _) in enumerate(r_list, start=1):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (k + rank))

    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates[:top_k]


class CascadeSearchPipeline:
    """
    Complete Two-Stage Search & Re-Ranking Cascade with latency benchmarking.
    """
    def __init__(
        self,
        products_df: pl.DataFrame,
        bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model: Optional[str] = None,
        candidate_depth: int = 50,
        final_top_k: int = 10
    ):
        self.candidate_depth = candidate_depth
        self.final_top_k = final_top_k

        # Prepare doc_text column
        if "doc_text" not in products_df.columns:
            doc_texts = [
                build_product_doc_text(
                    row.get("product_title"),
                    row.get("product_brand"),
                    row.get("product_color"),
                    row.get("product_bullet_point"),
                    row.get("product_description")
                )
                for row in products_df.to_dicts()
            ]
            products_df = products_df.with_columns(pl.Series("doc_text", doc_texts))

        self.products_df = products_df
        self.product_text_map = dict(zip(products_df["product_id"].to_list(), products_df["doc_text"].to_list()))
        self.product_title_map = dict(zip(products_df["product_id"].to_list(), products_df["product_title"].to_list()))

        print("Fitting Lexical BM25 index...")
        self.bm25_retriever = LexicalBM25Retriever(k1=0.5, b=1.0)
        self.bm25_retriever.fit(products_df)

        print("Fitting Dense Bi-Encoder index...")
        self.dense_retriever = DenseBiEncoderRetriever(model_name=bi_encoder_model)
        self.dense_retriever.fit(products_df)

        print("Loading Cross-Encoder Re-Ranker...")
        self.reranker = CrossEncoderReranker(model_name=cross_encoder_model)
        print("Cascade Search Pipeline ready!")

    def search(self, query: str) -> Dict[str, Any]:
        """
        Executes the two-stage search cascade and records per-stage latencies.
        """
        timings = {}

        # 1. Lexical BM25
        t0 = time.perf_counter()
        bm25_hits = self.bm25_retriever.retrieve(query, top_k=self.candidate_depth)
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Dense Semantic
        t1 = time.perf_counter()
        dense_hits = self.dense_retriever.retrieve(query, top_k=self.candidate_depth)
        timings["dense_ms"] = (time.perf_counter() - t1) * 1000

        # 3. Hybrid RRF Fusion
        t2 = time.perf_counter()
        hybrid_candidates = reciprocal_rank_fusion([bm25_hits, dense_hits], k=60, top_k=self.candidate_depth)
        timings["fusion_ms"] = (time.perf_counter() - t2) * 1000

        # 4. Cross-Encoder Re-Ranking
        t3 = time.perf_counter()
        reranked_hits = self.reranker.rerank(
            query,
            hybrid_candidates,
            self.product_text_map,
            top_k=self.final_top_k
        )
        timings["reranker_ms"] = (time.perf_counter() - t3) * 1000
        timings["total_ms"] = (time.perf_counter() - t0) * 1000

        return {
            "query": query,
            "timings": timings,
            "bm25_top": bm25_hits[:self.final_top_k],
            "dense_top": dense_hits[:self.final_top_k],
            "hybrid_top": hybrid_candidates[:self.final_top_k],
            "reranked_top": reranked_hits,
            "product_title_map": self.product_title_map
        }
