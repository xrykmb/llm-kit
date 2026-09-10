from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import ChatClient, chat_completion
from .eval import evaluate_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-kit", description="OpenAI-compatible chat and JSONL eval.")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="Send one user message")
    chat.add_argument("prompt", help="User prompt")
    chat.add_argument("--system", default="", help="Optional system prompt")

    eval_cmd = sub.add_parser("eval", help="Run JSONL substring eval")
    eval_cmd.add_argument("path", help="Path to .jsonl file")
    eval_cmd.add_argument("--json", action="store_true", help="Print JSON results")

    args = parser.parse_args(argv)
    client = ChatClient.from_env()

    if args.command == "chat":
        messages: list[dict[str, str]] = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        print(chat_completion(client, messages))
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
