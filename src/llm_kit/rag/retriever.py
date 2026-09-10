from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .chunker import Chunk

_TOKEN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    words = _TOKEN.findall(lowered)
    compact = re.sub(r"\s+", "", lowered)
    grams = [compact[i : i + 2] for i in range(max(0, len(compact) - 1))]
    return words + grams


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class TfIdfRetriever:
    """Lexical retriever. Swap this class to plug in embeddings later."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks: list[Chunk] = list(chunks or [])
        self._tf: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        self._rebuild()

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._rebuild()

    def search(self, query: str, k: int = 4) -> list[ScoredChunk]:
        if k <= 0 or not self.chunks:
            return []
        q = Counter(tokenize(query))
        if not q:
            return []
        scored: list[ScoredChunk] = []
        q_vec = self._weighted(q)
        q_norm = _norm(q_vec)
        for chunk, tf in zip(self.chunks, self._tf):
            d_vec = self._weighted(tf)
            score = _cosine(q_vec, d_vec, q_norm, _norm(d_vec))
            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]

    def _rebuild(self) -> None:
        self._tf = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = max(len(self.chunks), 1)
        self._idf = {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}

    def _weighted(self, tf: Counter[str]) -> dict[str, float]:
        return {term: (1 + math.log(count)) * self._idf.get(term, 1.0) for term, count in tf.items() if count > 0}


def _norm(vec: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vec.values())) or 1.0


def _cosine(a: dict[str, float], b: dict[str, float], a_norm: float, b_norm: float) -> float:
    keys = a.keys() & b.keys()
    if not keys:
        return 0.0
    return sum(a[k] * b[k] for k in keys) / (a_norm * b_norm)
