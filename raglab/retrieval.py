from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def dense_search(query_vector: list[float], chunks: list[dict], top_k: int = 5) -> list[dict]:
    results = []
    for chunk in chunks:
        vector = chunk.get("embedding")
        if vector:
            results.append({**chunk, "score": _cosine_similarity(query_vector, vector), "retrieval": "dense"})
    return sorted(results, key=lambda item: item["score"], reverse=True)[:max(0, top_k)]


def bm25_scores(query: str, chunks: list[dict], k1: float = 1.5, b: float = 0.75) -> list[float]:
    documents = [tokenize(item.get("text", "")) for item in chunks]
    if not documents:
        return []
    lengths = [len(doc) for doc in documents]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    doc_freq = Counter(token for doc in documents for token in set(doc))
    query_terms = tokenize(query)
    scores = []
    n_docs = len(documents)
    for doc, doc_len in zip(documents, lengths):
        frequencies = Counter(doc)
        score = 0.0
        for term in query_terms:
            freq = frequencies.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denominator = freq + k1 * (1 - b + b * doc_len / (average_length or 1))
            score += idf * (freq * (k1 + 1)) / denominator
        scores.append(score)
    return scores


def bm25_search(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    scores = bm25_scores(query, chunks)
    results = [{**chunk, "score": score, "retrieval": "bm25"} for chunk, score in zip(chunks, scores)]
    return sorted(results, key=lambda item: item["score"], reverse=True)[:max(0, top_k)]


def reciprocal_rank_fusion(*ranked_lists: list[dict], k: int = 60) -> list[dict]:
    """Fuse ranked lists by stable chunk_id using reciprocal rank fusion."""
    fused: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = str(item.get("chunk_id") or f"{item.get('source')}:{item.get('text', '')[:80]}")
            if key not in fused:
                fused[key] = {**item, "score": 0.0, "retrieval": "hybrid"}
            fused[key]["score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)


def hybrid_search(query: str, query_vector: list[float], chunks: list[dict], top_k: int = 5) -> list[dict]:
    candidate_k = max(top_k * 3, 12)
    dense = dense_search(query_vector, chunks, candidate_k)
    lexical = bm25_search(query, chunks, candidate_k)
    return reciprocal_rank_fusion(dense, lexical)[:max(0, top_k)]


def rerank_results(query: str, results: list[dict], top_k: int = 5, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> list[dict]:
    """Optional cross-encoder reranking. Model downloads on first use."""
    if not results:
        return []
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    pairs = [(query, item.get("text", "")) for item in results]
    scores = model.predict(pairs)
    reranked = [{**item, "score": float(score), "retrieval": "cross-encoder"} for item, score in zip(results, scores)]
    return sorted(reranked, key=lambda item: item["score"], reverse=True)[:max(0, top_k)]
