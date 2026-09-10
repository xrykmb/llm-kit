from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .client import ChatClient, ChatResult, collect_stream, complete
from .eval import evaluate_jsonl
from .rag.pipeline import ask, ingest
from .session import Conversation
from .stream import StreamEvent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-kit",
        description="DeepSeek RAG + CoT toolkit (OpenAI-compatible API).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="Chat (omit prompt for multi-turn REPL)")
    chat.add_argument("prompt", nargs="?", help="User prompt; omit to enter REPL")
    chat.add_argument("--system", default="", help="Optional system prompt")
    chat.add_argument("--show-reasoning", action="store_true", help="Print DeepSeek reasoning_content")
    chat.add_argument("--stream", dest="stream", action="store_true", default=True)
    chat.add_argument("--no-stream", dest="stream", action="store_false")
    chat.add_argument("--session", default="", help="JSON file to load/save conversation")
    chat.add_argument("--max-turns", type=int, default=12, help="Keep the last N user/assistant turns")

    eval_cmd = sub.add_parser("eval", help="Run JSONL substring eval")
    eval_cmd.add_argument("path", help="Path to .jsonl file")
    eval_cmd.add_argument("--json", action="store_true", help="Print JSON results")

    ingest_cmd = sub.add_parser("ingest", help="Build a RAG index from files or a directory")
    ingest_cmd.add_argument("path", help="File or directory (.txt / .md)")
    ingest_cmd.add_argument("--index", default=".llm-kit/index.json", help="Index JSON path")
    ingest_cmd.add_argument("--size", type=int, default=400, help="Chunk size in characters")
    ingest_cmd.add_argument("--overlap", type=int, default=80, help="Chunk overlap")
    ingest_cmd.add_argument("--mmr-lambda", type=float, default=0.7, help="MMR λ (relevance vs diversity)")

    rag = sub.add_parser("rag", help="RAG + CoT (omit question for multi-turn REPL)")
    rag.add_argument("question", nargs="?", help="User question; omit to enter REPL")
    rag.add_argument("--index", default=".llm-kit/index.json", help="Index JSON path")
    rag.add_argument("-k", type=int, default=4, help="Top-k chunks")
    rag.add_argument("--no-cot", action="store_true", help="Disable Chain-of-Thought prompt")
    rag.add_argument("--json", action="store_true", help="Print JSON")
    rag.add_argument("--show-reasoning", action="store_true", help="Print reasoning text")
    rag.add_argument("--stream", dest="stream", action="store_true", default=True)
    rag.add_argument("--no-stream", dest="stream", action="store_false")
    rag.add_argument("--session", default="", help="JSON file to load/save conversation")
    rag.add_argument("--max-turns", type=int, default=12)

    args = parser.parse_args(argv)
    client = ChatClient.from_env()

    if args.command == "chat":
        conversation = _load_conversation(args.session, args.system, args.max_turns)
        if args.prompt is None:
            return _chat_repl(client, conversation, args)
        _chat_turn(client, conversation, args.prompt, args)
        _maybe_save(conversation, args.session)
        return 0

    if args.command == "ingest":
        count = ingest(
            [Path(args.path)],
            Path(args.index),
            size=args.size,
            overlap=args.overlap,
            mmr_lambda=args.mmr_lambda,
        )
        print(f"llm-kit ingest: {count} chunks -> {args.index}")
        return 0 if count else 1

    if args.command == "rag":
        conversation = _load_conversation(args.session, "", args.max_turns)
        if args.question is None:
            return _rag_repl(client, conversation, args)
        answer = _rag_turn(client, conversation, args.question, args)
        _maybe_save(conversation, args.session)
        return _print_rag(answer, args)

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


def _load_conversation(session: str, system: str, max_turns: int) -> Conversation:
    if session and Path(session).exists():
        convo = Conversation.load(Path(session))
        if system:
            convo.system = system
        convo.max_turns = max_turns
        return convo
    return Conversation(system=system, max_turns=max_turns)


def _maybe_save(conversation: Conversation, session: str) -> None:
    if session:
        conversation.save(Path(session))


def _on_event(show_reasoning: bool) -> tuple[list[bool], object]:
    started = [False]

    def handler(event: StreamEvent) -> None:
        if event.kind == "reasoning":
            if show_reasoning:
                print(event.text, end="", flush=True)
            return
        if show_reasoning and not started[0]:
            print("\n---", flush=True)
            started[0] = True
        print(event.text, end="", flush=True)

    return started, handler


def _chat_turn(client: ChatClient, conversation: Conversation, prompt: str, args) -> ChatResult:
    conversation.add_user(prompt)
    messages = conversation.to_api_messages()
    if args.stream:
        _, handler = _on_event(args.show_reasoning)
        result = collect_stream(client, messages, on_event=handler)
        print()
    else:
        result = complete(client, messages)
        if args.show_reasoning and result.reasoning:
            print(result.reasoning)
            print("---")
        print(result.content)
    conversation.add_assistant(result.content)
    return result


def _rag_turn(client: ChatClient, conversation: Conversation, question: str, args):
    _, handler = _on_event(args.show_reasoning)
    return ask(
        question,
        Path(args.index),
        client,
        k=args.k,
        cot=not args.no_cot,
        conversation=conversation,
        stream=args.stream and not args.json,
        on_event=handler if (args.stream and not args.json) else None,
    )


def _print_rag(answer, args) -> int:
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
    if not args.stream:
        print(answer.answer)
        if args.show_reasoning and answer.reasoning:
            print("\n思考：")
            print(answer.reasoning)
    else:
        print()
    if answer.hits:
        print("\n依据：")
        for hit in answer.hits:
            print(f"  - {hit.chunk.id} ({hit.score:.3f}) {hit.chunk.source}")
    return 0


def _chat_repl(client: ChatClient, conversation: Conversation, args) -> int:
    print("llm-kit chat  输入 /exit 退出，/reset 清空，/save 保存会话", file=sys.stderr)
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        text = line.strip()
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            break
        if text == "/reset":
            conversation.reset()
            print("已清空会话", file=sys.stderr)
            continue
        if text == "/save" or text.startswith("/save "):
            path = text.split(" ", 1)[1].strip() if " " in text else args.session
            if not path:
                print("用法: /save 路径.json", file=sys.stderr)
                continue
            conversation.save(Path(path))
            print(f"已保存 {path}", file=sys.stderr)
            continue
        print("bot> ", end="", flush=True)
        _chat_turn(client, conversation, text, args)
    _maybe_save(conversation, args.session)
    return 0


def _rag_repl(client: ChatClient, conversation: Conversation, args) -> int:
    print("llm-kit rag  输入 /exit 退出，/reset 清空", file=sys.stderr)
    args.json = False
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        text = line.strip()
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            break
        if text == "/reset":
            conversation.reset()
            print("已清空会话", file=sys.stderr)
            continue
        print("bot> ", end="", flush=True)
        answer = _rag_turn(client, conversation, text, args)
        _print_rag(answer, args)
    _maybe_save(conversation, args.session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
