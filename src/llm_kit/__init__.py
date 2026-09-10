"""OpenAI-compatible LLM client and JSONL eval helpers."""

from .client import ChatClient, chat_completion
from .eval import EvalCase, EvalResult, evaluate_jsonl

__all__ = [
    "ChatClient",
    "EvalCase",
    "EvalResult",
    "chat_completion",
    "evaluate_jsonl",
]
__version__ = "0.1.0"
