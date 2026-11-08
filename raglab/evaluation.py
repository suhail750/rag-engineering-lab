from __future__ import annotations


def evaluate_retrieval(rows: list[dict], k: int = 5) -> dict[str, float]:
    """Calculate source-level Hit Rate@k, Recall@k, and MRR@k."""
    if not rows:
        return {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0}
    hits = 0
    recalls = []
    reciprocal_ranks = []
    for row in rows:
        relevant = {str(value) for value in row.get("relevant_sources", [])}
        retrieved = [str(value) for value in row.get("retrieved_sources", [])[:k]]
        matched = relevant.intersection(retrieved)
        hits += int(bool(matched))
        recalls.append(len(matched) / len(relevant) if relevant else 0.0)
        rank = next((i for i, source in enumerate(retrieved, start=1) if source in relevant), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    count = len(rows)
    return {
        "hit_rate": hits / count,
        "recall": sum(recalls) / count,
        "mrr": sum(reciprocal_ranks) / count,
    }
