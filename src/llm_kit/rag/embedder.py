from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

from .tokens import tokenize

DEFAULT_EMBED_DIM = 384


class Embedder(Protocol):
    name: str

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Vectors should already be L2-normalized."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class HashingEmbedder:
    """Deterministic dense vectors via signed feature hashing (offline / tests)."""

    name = "hashing"

    def __init__(self, dim: int = DEFAULT_EMBED_DIM) -> None:
        self.dim = dim

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        return l2_normalize(vector)


class OpenAICompatEmbedder:
    """POST /embeddings — OpenAI, SiliconFlow, Ollama, etc. (DeepSeek has no embed API)."""

    name = "openai-compat"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        timeout_sec: float = 120.0,
        opener: Any = urllib.request.urlopen,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec
        self._opener = opener

    @classmethod
    def from_env(cls, opener: Any = urllib.request.urlopen) -> OpenAICompatEmbedder:
        return cls(
            api_key=os.environ.get("LLM_EMBED_API_KEY") or os.environ.get("LLM_API_KEY", ""),
            base_url=os.environ.get("LLM_EMBED_BASE_URL", "").rstrip("/"),
            model=os.environ.get("LLM_EMBED_MODEL", "text-embedding-3-small"),
            opener=opener,
        )

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), 32):
            out.extend(self._embed_batch(texts[start : start + 32]))
        return out

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embedding HTTP {exc.code}: {body}") from exc
        data = json.loads(raw)
        try:
            rows = list(data["data"])
            if all(isinstance(row, dict) and "index" in row for row in rows):
                rows = sorted(rows, key=lambda item: item["index"])
            vectors = [l2_normalize([float(x) for x in row["embedding"]]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Unexpected embedding response: {data!r}") from exc
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding count does not match input texts")
        return vectors


def embedder_from_env() -> Embedder:
    base = os.environ.get("LLM_EMBED_BASE_URL", "").strip()
    if base:
        return OpenAICompatEmbedder.from_env()
    return HashingEmbedder()
