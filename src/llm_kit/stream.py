from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

Kind = Literal["reasoning", "content"]


@dataclass(frozen=True)
class StreamEvent:
    kind: Kind
    text: str


def iter_sse_data_lines(lines: Iterable[str]) -> Iterator[str]:
    for raw in lines:
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        yield line[5:].strip()


def iter_sse_payloads(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for data in iter_sse_data_lines(lines):
        if data == "[DONE]":
            return
        yield json.loads(data)


def events_from_payload(payload: dict[str, Any]) -> list[StreamEvent]:
    try:
        delta = payload["choices"][0].get("delta") or {}
    except (KeyError, IndexError, TypeError):
        return []
    events: list[StreamEvent] = []
    reasoning = delta.get("reasoning_content")
    content = delta.get("content")
    if isinstance(reasoning, str) and reasoning:
        events.append(StreamEvent("reasoning", reasoning))
    if isinstance(content, str) and content:
        events.append(StreamEvent("content", content))
    return events
