from __future__ import annotations

import json
from pathlib import Path

from .chunker import Chunk
from .embedder import HashingEmbedder, OpenAICompatEmbedder
from .retriever import HybridRetriever


def save_index(path: Path, retriever: HybridRetriever) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "embedder": getattr(retriever.embedder, "name", "hashing"),
        "model": getattr(retriever.embedder, "model", ""),
        "mmr_lambda": retriever.mmr_lambda,
        "chunks": [
            {
                "id": chunk.id,
                "source": chunk.source,
                "text": chunk.text,
                "vector": vector,
            }
            for chunk, vector in zip(retriever.chunks, retriever.vectors)
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_index(path: Path, embedder=None) -> HybridRetriever:
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for row in data.get("chunks") or []:
        chunks.append(Chunk(id=str(row["id"]), source=str(row["source"]), text=str(row["text"])))
        if row.get("vector"):
            vectors.append([float(x) for x in row["vector"]])
    kind = data.get("embedder") or "hashing"
    if embedder is None:
        if kind == "openai-compat":
            embedder = OpenAICompatEmbedder.from_env()
        else:
            embedder = HashingEmbedder()
    if len(vectors) != len(chunks):
        vectors = embedder.embed_many([chunk.text for chunk in chunks]) if chunks else []
    return HybridRetriever(
        chunks,
        vectors,
        embedder=embedder,
        mmr_lambda=float(data.get("mmr_lambda") or 0.7),
    )
