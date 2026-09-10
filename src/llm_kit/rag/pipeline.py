from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..client import ChatClient, ChatResult, complete
from ..cot import build_rag_messages, extract_final_answer
from .chunker import chunk_documents
from .loader import load_paths
from .retriever import ScoredChunk, TfIdfRetriever
from .store import load_index, save_index


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    reasoning: str
    hits: tuple[ScoredChunk, ...]
    raw: str


def ingest(paths: list[Path], index_path: Path, *, size: int = 400, overlap: int = 80) -> int:
    documents = load_paths(paths)
    chunks = chunk_documents(documents, size=size, overlap=overlap)
    retriever = TfIdfRetriever(chunks)
    save_index(index_path, retriever)
    return len(chunks)


def ask(
    question: str,
    index_path: Path,
    client: ChatClient,
    *,
    k: int = 4,
    cot: bool = True,
    complete_fn: Callable[..., ChatResult] = complete,
) -> RagAnswer:
    retriever = load_index(index_path)
    hits = retriever.search(question, k=k)
    contexts = [hit.chunk.text for hit in hits]
    messages = build_rag_messages(question, contexts, cot=cot)
    result = complete_fn(client, messages)
    answer = extract_final_answer(result.content) if cot else result.content.strip()
    reasoning = result.reasoning or (
        result.content.split("最终答案：", 1)[0].replace("思考：", "").strip() if cot and "最终答案：" in result.content else ""
    )
    return RagAnswer(
        question=question,
        answer=answer,
        reasoning=reasoning,
        hits=tuple(hits),
        raw=result.content,
    )
