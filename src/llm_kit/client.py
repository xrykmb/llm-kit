from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .stream import StreamEvent, events_from_payload, iter_sse_payloads

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


@dataclass(frozen=True)
class ChatClient:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_sec: float = 120.0

    @classmethod
    def from_env(cls) -> ChatClient:
        return cls(
            api_key=os.environ.get("LLM_API_KEY", ""),
            base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        )


@dataclass(frozen=True)
class ChatResult:
    content: str
    reasoning: str = ""


def _payload(
    client: ChatClient,
    messages: list[dict[str, str]],
    *,
    temperature: float | None,
    stream: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": client.model,
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None and "reasoner" not in client.model:
        body["temperature"] = temperature
    return body


def _headers(client: ChatClient, *, stream: bool) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if stream:
        headers["Accept"] = "text/event-stream"
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"
    return headers


def _open(client: ChatClient, payload: dict[str, Any], headers: dict[str, str], opener: Any):
    url = f"{client.base_url.rstrip('/')}/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        return opener(request, timeout=client.timeout_sec)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {body}") from exc


def complete(
    client: ChatClient,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = 0.2,
    opener: Any = urllib.request.urlopen,
) -> ChatResult:
    """Non-streaming POST /chat/completions."""
    response = _open(
        client,
        _payload(client, messages, temperature=temperature, stream=False),
        _headers(client, stream=False),
        opener,
    )
    with response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    try:
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {data!r}") from exc
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise RuntimeError(f"Unexpected content types in: {message!r}")
    return ChatResult(content=content, reasoning=reasoning)


def complete_stream(
    client: ChatClient,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = 0.2,
    opener: Any = urllib.request.urlopen,
) -> Iterator[StreamEvent]:
    """Yield content/reasoning deltas from an SSE stream."""
    response = _open(
        client,
        _payload(client, messages, temperature=temperature, stream=True),
        _headers(client, stream=True),
        opener,
    )
    with response:
        def lines() -> Iterator[str]:
            while True:
                raw = response.readline()
                if not raw:
                    break
                if isinstance(raw, bytes):
                    yield raw.decode("utf-8")
                else:
                    yield str(raw)

        for payload in iter_sse_payloads(lines()):
            yield from events_from_payload(payload)


def collect_stream(
    client: ChatClient,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = 0.2,
    opener: Any = urllib.request.urlopen,
    on_event: Any = None,
) -> ChatResult:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for event in complete_stream(client, messages, temperature=temperature, opener=opener):
        if on_event:
            on_event(event)
        if event.kind == "content":
            content_parts.append(event.text)
        else:
            reasoning_parts.append(event.text)
    return ChatResult(content="".join(content_parts), reasoning="".join(reasoning_parts))


def chat_completion(
    client: ChatClient,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = 0.2,
    opener: Any = urllib.request.urlopen,
) -> str:
    return complete(client, messages, temperature=temperature, opener=opener).content
