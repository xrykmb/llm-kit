# Contributing

1. Keep the runtime dependency-free (stdlib only).
2. Tests must not call a real LLM or embedding API.
3. Keep RAG stages separate: `loader` → `chunker` → `embedder` → hybrid `retriever` → `pipeline`.
4. Run `python -m pytest` before opening a pull request.
