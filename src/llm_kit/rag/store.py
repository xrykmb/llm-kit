from __future__ import annotations

import json
from pathlib import Path

from .chunker import Chunk
from .retriever import TfIdfRetriever


def save_index(path: Path, retriever: TfIdfRetriever) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunks": [
            {"id": chunk.id, "source": chunk.source, "text": chunk.text}
            for chunk in retriever.chunks
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(path: Path) -> TfIdfRetriever:
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = [
        Chunk(id=str(row["id"]), source=str(row["source"]), text=str(row["text"]))
        for row in data.get("chunks") or []
    ]
    return TfIdfRetriever(chunks)
