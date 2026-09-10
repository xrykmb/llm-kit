"""Modular RAG pieces: load → chunk → embed → hybrid retrieve → generate."""

from .chunker import Chunk, chunk_documents
from .embedder import HashingEmbedder, OpenAICompatEmbedder, embedder_from_env
from .loader import Document, load_paths
from .pipeline import RagAnswer, ask, ingest
from .retriever import HybridRetriever, ScoredChunk, TfIdfRetriever
from .store import load_index, save_index

__all__ = [
    "Chunk",
    "Document",
    "HashingEmbedder",
    "HybridRetriever",
    "OpenAICompatEmbedder",
    "RagAnswer",
    "ScoredChunk",
    "TfIdfRetriever",
    "ask",
    "chunk_documents",
    "embedder_from_env",
    "ingest",
    "load_index",
    "load_paths",
    "save_index",
]
