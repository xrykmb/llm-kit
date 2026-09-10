from __future__ import annotations

from dataclasses import dataclass

from .loader import Document


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str


def chunk_documents(
    documents: list[Document],
    *,
    size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[Chunk] = []
    for doc_index, document in enumerate(documents):
        text = " ".join(document.text.split())
        if not text:
            continue
        start = 0
        part = 0
        while start < len(text):
            piece = text[start : start + size]
            chunks.append(
                Chunk(
                    id=f"{doc_index}-{part}",
                    source=document.source,
                    text=piece,
                )
            )
            part += 1
            start += size - overlap
    return chunks
