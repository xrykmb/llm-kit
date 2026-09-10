from __future__ import annotations

from .embedder import cosine


def reciprocal_rank_fusion(
    ranked_ids: list[list[str]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """RRF: score(d) = sum 1 / (k + rank)."""
    fused: dict[str, float] = {}
    for ranking in ranked_ids:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


def maximal_marginal_relevance(
    query_vec: list[float],
    candidate_ids: list[str],
    vectors: dict[str, list[float]],
    *,
    k: int,
    lambda_: float = 0.7,
) -> list[str]:
    """MMR: λ * sim(q, d) - (1-λ) * max sim(d, selected)."""
    remaining = [item_id for item_id in candidate_ids if item_id in vectors]
    selected: list[str] = []
    while remaining and len(selected) < k:
        best_id = remaining[0]
        best_score = float("-inf")
        for item_id in remaining:
            relevance = cosine(query_vec, vectors[item_id])
            redundancy = 0.0
            if selected:
                redundancy = max(cosine(vectors[item_id], vectors[other]) for other in selected)
            score = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_id = item_id
        selected.append(best_id)
        remaining.remove(best_id)
    return selected
