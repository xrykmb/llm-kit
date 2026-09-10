"""Modular RAG pieces: load → chunk → retrieve → generate."""

from .chunker import Chunk, chunk_documents
from .loader import Document, load_paths
from .pipeline import RagAnswer, ask, ingest
from .retriever import ScoredChunk, TfIdfRetriever
from .store import load_index, save_index

__all__ = [
    "Chunk",
    "Document",
    "RagAnswer",
    "ScoredChunk",
    "TfIdfRetriever",
    "ask",
    "chunk_documents",
    "ingest",
    "load_index",
    "load_paths",
    "save_index",
]
