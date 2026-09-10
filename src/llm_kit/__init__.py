"""DeepSeek RAG + CoT toolkit."""

from .client import ChatClient, ChatResult, chat_completion, complete
from .cot import build_rag_messages, extract_final_answer
from .eval import EvalCase, EvalResult, evaluate_jsonl
from .rag import HybridRetriever, RagAnswer, ask, ingest

__all__ = [
    "ChatClient",
    "ChatResult",
    "EvalCase",
    "EvalResult",
    "HybridRetriever",
    "RagAnswer",
    "ask",
    "build_rag_messages",
    "chat_completion",
    "complete",
    "evaluate_jsonl",
    "extract_final_answer",
    "ingest",
]
__version__ = "0.3.0"
