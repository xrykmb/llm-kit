from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import ChatClient, complete
from .eval import evaluate_jsonl
from .rag.pipeline import ask, ingest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-kit",
        description="DeepSeek RAG + CoT toolkit (OpenAI-compatible API).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="Send one user message")
    chat.add_argument("prompt", help="User prompt")
    chat.add_argument("--system", default="", help="Optional system prompt")
    chat.add_argument("--show-reasoning", action="store_true", help="Print DeepSeek reasoning_content")

    eval_cmd = sub.add_parser("eval", help="Run JSONL substring eval")
    eval_cmd.add_argument("path", help="Path to .jsonl file")
    eval_cmd.add_argument("--json", action="store_true", help="Print JSON results")

    ingest_cmd = sub.add_parser("ingest", help="Build a RAG index from files or a directory")
    ingest_cmd.add_argument("path", help="File or directory (.txt / .md)")
    ingest_cmd.add_argument("--index", default=".llm-kit/index.json", help="Index JSON path")
    ingest_cmd.add_argument("--size", type=int, default=400, help="Chunk size in characters")
    ingest_cmd.add_argument("--overlap", type=int, default=80, help="Chunk overlap")

    rag = sub.add_parser("rag", help="Retrieve then answer with CoT on DeepSeek")
    rag.add_argument("question", help="User question")
    rag.add_argument("--index", default=".llm-kit/index.json", help="Index JSON path")
    rag.add_argument("-k", type=int, default=4, help="Top-k chunks")
    rag.add_argument("--no-cot", action="store_true", help="Disable Chain-of-Thought prompt")
    rag.add_argument("--json", action="store_true", help="Print JSON")
    rag.add_argument("--show-reasoning", action="store_true", help="Print reasoning text")

    args = parser.parse_args(argv)
    client = ChatClient.from_env()

    if args.command == "chat":
        messages: list[dict[str, str]] = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        result = complete(client, messages)
        if args.show_reasoning and result.reasoning:
            print(result.reasoning)
            print("---")
        print(result.content)
        return 0

    if args.command == "ingest":
        count = ingest([Path(args.path)], Path(args.index), size=args.size, overlap=args.overlap)
        print(f"llm-kit ingest: {count} chunks -> {args.index}")
        return 0 if count else 1

    if args.command == "rag":
        answer = ask(
            args.question,
            Path(args.index),
            client,
            k=args.k,
            cot=not args.no_cot,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "answer": answer.answer,
                        "reasoning": answer.reasoning,
                        "raw": answer.raw,
                        "hits": [
                            {
                                "id": hit.chunk.id,
                                "source": hit.chunk.source,
                                "score": round(hit.score, 4),
                                "text": hit.chunk.text,
                            }
                            for hit in answer.hits
                        ],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        print(answer.answer)
        if args.show_reasoning and answer.reasoning:
            print("\n思考：")
            print(answer.reasoning)
        if answer.hits:
            print("\n依据：")
            for hit in answer.hits:
                print(f"  - {hit.chunk.id} ({hit.score:.3f}) {hit.chunk.source}")
        return 0

    results = evaluate_jsonl(Path(args.path), client)
    passed = sum(1 for item in results if item.passed)
    payload = [
        {
            "id": item.case_id,
            "passed": item.passed,
            "missing": list(item.missing),
            "output": item.output,
        }
        for item in results
    ]
    if args.json:
        print(json.dumps({"passed": passed, "total": len(results), "cases": payload}, ensure_ascii=False))
    else:
        print(f"llm-kit eval: {passed}/{len(results)} passed")
        for item in results:
            mark = "ok" if item.passed else "fail"
            extra = f" missing={list(item.missing)}" if item.missing else ""
            print(f"  [{mark}] {item.case_id}{extra}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
