from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".txt", ".md", ".markdown"}


@dataclass(frozen=True)
class Document:
    source: str
    text: str


def load_paths(paths: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED:
                    documents.append(_read(child))
        elif path.is_file() and path.suffix.lower() in SUPPORTED:
            documents.append(_read(path))
    return documents


def _read(path: Path) -> Document:
    return Document(source=str(path.resolve()), text=path.read_text(encoding="utf-8"))
