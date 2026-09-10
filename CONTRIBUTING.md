# Contributing

1. Keep the runtime dependency-free (stdlib only).
2. Tests must not call a real LLM API.
3. Keep RAG stages separate: `loader` → `chunker` → `retriever` → `pipeline`.
4. Run `python -m pytest` before opening a pull request.
