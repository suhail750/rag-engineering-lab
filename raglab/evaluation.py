from __future__ import annotations

import math


def evaluate_retrieval(rows: list[dict], k: int = 5) -> dict[str, float]:
    """Calculate source-level Hit Rate@k, Recall@k, Precision@k, MRR@k, and nDCG@k."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not rows:
        return {"hit_rate": 0.0, "recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}

    hit_count = 0
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for row in rows:
        relevant = {str(value) for value in row.get("relevant_sources", [])}
        retrieved = [str(value) for value in row.get("retrieved_sources", [])[:k]]
        if not relevant:
            recalls.append(0.0)
            precisions.append(0.0)
            reciprocal_ranks.append(0.0)
            ndcgs.append(0.0)
            continue

        matched = relevant.intersection(retrieved)
        hit_count += int(bool(matched))
        recalls.append(len(matched) / len(relevant))
        precisions.append(len(matched) / k)

        first_rank = next((rank for rank, source in enumerate(retrieved, start=1) if source in relevant), None)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

        dcg = sum(1.0 / math.log2(rank + 1) for rank, source in enumerate(retrieved, start=1) if source in relevant)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    count = len(rows)
    return {
        "hit_rate": hit_count / count,
        "recall": sum(recalls) / count,
        "precision": sum(precisions) / count,
        "mrr": sum(reciprocal_ranks) / count,
        "ndcg": sum(ndcgs) / count,
    }
