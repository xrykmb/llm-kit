from __future__ import annotations

import math
from collections import Counter

from .chunker import Chunk
from .tokens import tokenize


class BM25Index:
    """Okapi BM25 lexical scores (k1=1.5, b=0.75)."""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tf = [Counter(tokenize(chunk.text)) for chunk in chunks]
        lengths = [sum(tf.values()) or 1 for tf in self._tf]
        self._avgdl = (sum(lengths) / len(lengths)) if lengths else 1.0
        self._dl = lengths
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = max(len(chunks), 1)
        self._idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def scores(self, query: str) -> list[float]:
        qf = Counter(tokenize(query))
        out = [0.0] * len(self._tf)
        if not qf:
            return out
        for i, tf in enumerate(self._tf):
            score = 0.0
            dl = self._dl[i]
            for term, q_count in qf.items():
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += q_count * idf * (freq * (self.k1 + 1)) / denom
            out[i] = score
        return out
