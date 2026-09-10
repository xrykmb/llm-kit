"""DeepSeek RAG + CoT toolkit."""

from .client import ChatClient, ChatResult, chat_completion, collect_stream, complete, complete_stream
from .cot import build_rag_messages, extract_final_answer
from .eval import EvalCase, EvalResult, evaluate_jsonl
from .rag import HybridRetriever, RagAnswer, ask, ingest
from .session import Conversation
from .stream import StreamEvent

__all__ = [
    "ChatClient",
    "ChatResult",
    "Conversation",
    "EvalCase",
    "EvalResult",
    "HybridRetriever",
    "RagAnswer",
    "StreamEvent",
    "ask",
    "build_rag_messages",
    "chat_completion",
    "collect_stream",
    "complete",
    "complete_stream",
    "evaluate_jsonl",
    "extract_final_answer",
    "ingest",
]
__version__ = "0.4.0"
