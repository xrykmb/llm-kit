from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25Index
from .chunker import Chunk
from .embedder import Embedder, HashingEmbedder, cosine
from .similarity import maximal_marginal_relevance, reciprocal_rank_fusion
from .tokens import tokenize

__all__ = ["ScoredChunk", "HybridRetriever", "TfIdfRetriever", "tokenize"]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class HybridRetriever:
    """Dense cosine + BM25, fused with RRF, then MMR for diversity."""

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        *,
        embedder: Embedder | None = None,
        rrf_k: int = 60,
        mmr_lambda: float = 0.7,
        candidate_multiplier: int = 5,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        self.chunks = list(chunks)
        self.vectors = list(vectors)
        self.embedder = embedder or HashingEmbedder()
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda
        self.candidate_multiplier = max(candidate_multiplier, 1)
        self._bm25 = BM25Index(self.chunks)
        self._by_id = {chunk.id: chunk for chunk in self.chunks}
        self._vec_by_id = {chunk.id: vector for chunk, vector in zip(self.chunks, self.vectors)}

    def search(self, query: str, k: int = 4) -> list[ScoredChunk]:
        if k <= 0 or not self.chunks:
            return []
        query_vec = self.embedder.embed_many([query])[0]
        dense_ranked = self._dense_ranking(query_vec)
        lexical_ranked = self._bm25_ranking(query)
        fused = reciprocal_rank_fusion([dense_ranked, lexical_ranked], k=self.rrf_k)
        pool = max(k * self.candidate_multiplier, k)
        fused_ids = [item_id for item_id, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]
        candidate_ids = fused_ids[:pool] or dense_ranked[:pool]
        chosen = maximal_marginal_relevance(
            query_vec,
            candidate_ids,
            self._vec_by_id,
            k=k,
            lambda_=self.mmr_lambda,
        )
        return [
            ScoredChunk(chunk=self._by_id[item_id], score=fused.get(item_id, cosine(query_vec, self._vec_by_id[item_id])))
            for item_id in chosen
        ]

    def _dense_ranking(self, query_vec: list[float]) -> list[str]:
        scored = [
            (chunk.id, cosine(query_vec, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [item_id for item_id, score in scored if score > 0] or [item_id for item_id, _ in scored]

    def _bm25_ranking(self, query: str) -> list[str]:
        scores = self._bm25.scores(query)
        ranked = sorted(
            ((chunk.id, score) for chunk, score in zip(self.chunks, scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        return [item_id for item_id, score in ranked if score > 0]


class TfIdfRetriever(HybridRetriever):
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        chunks = list(chunks or [])
        embedder = HashingEmbedder()
        vectors = embedder.embed_many([chunk.text for chunk in chunks]) if chunks else []
        super().__init__(chunks, vectors, embedder=embedder)
