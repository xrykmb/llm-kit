from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..client import ChatClient, ChatResult, collect_stream, complete
from ..cot import build_rag_messages, extract_final_answer
from ..session import Conversation
from .chunker import chunk_documents
from .embedder import Embedder, embedder_from_env
from .loader import load_paths
from .retriever import HybridRetriever, ScoredChunk
from .store import load_index, save_index


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    reasoning: str
    hits: tuple[ScoredChunk, ...]
    raw: str


def ingest(
    paths: list[Path],
    index_path: Path,
    *,
    size: int = 400,
    overlap: int = 80,
    embedder: Embedder | None = None,
    mmr_lambda: float = 0.7,
) -> int:
    documents = load_paths(paths)
    chunks = chunk_documents(documents, size=size, overlap=overlap)
    chosen = embedder or embedder_from_env()
    vectors = chosen.embed_many([chunk.text for chunk in chunks]) if chunks else []
    retriever = HybridRetriever(chunks, vectors, embedder=chosen, mmr_lambda=mmr_lambda)
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
    embedder: Embedder | None = None,
    conversation: Conversation | None = None,
    stream: bool = False,
    on_event: Callable[..., None] | None = None,
) -> RagAnswer:
    retriever = load_index(index_path, embedder=embedder)
    hits = retriever.search(question, k=k)
    contexts = [hit.chunk.text for hit in hits]
    rag_messages = build_rag_messages(question, contexts, cot=cot)
    if conversation is None:
        messages = rag_messages
    else:
        messages = [rag_messages[0], *conversation.messages, rag_messages[-1]]
    if stream:
        result = collect_stream(client, messages, on_event=on_event)
    else:
        result = complete_fn(client, messages)
    answer = extract_final_answer(result.content) if cot else result.content.strip()
    reasoning = result.reasoning or (
        result.content.split("最终答案：", 1)[0].replace("思考：", "").strip() if cot and "最终答案：" in result.content else ""
    )
    if conversation is not None:
        conversation.add_user(question)
        conversation.add_assistant(answer)
    return RagAnswer(
        question=question,
        answer=answer,
        reasoning=reasoning,
        hits=tuple(hits),
        raw=result.content,
    )
