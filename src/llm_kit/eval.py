from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .client import ChatClient, chat_completion


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    expect_contains: tuple[str, ...]
    system: str = ""


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    output: str
    missing: tuple[str, ...]


def load_jsonl(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        contains = row.get("expect_contains") or []
        if isinstance(contains, str):
            contains = [contains]
        cases.append(
            EvalCase(
                id=str(row.get("id") or f"case-{index}"),
                prompt=str(row["prompt"]),
                expect_contains=tuple(str(item) for item in contains),
                system=str(row.get("system") or ""),
            )
        )
    return cases


def evaluate_jsonl(
    path: Path,
    client: ChatClient,
    *,
    complete: Callable[..., str] = chat_completion,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in load_jsonl(path):
        messages: list[dict[str, str]] = []
        if case.system:
            messages.append({"role": "system", "content": case.system})
        messages.append({"role": "user", "content": case.prompt})
        output = complete(client, messages)
        missing = tuple(item for item in case.expect_contains if item not in output)
        results.append(
            EvalResult(
                case_id=case.id,
                passed=not missing,
                output=output,
                missing=missing,
            )
        )
    return results
