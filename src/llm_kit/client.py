from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

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


def complete(
    client: ChatClient,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = 0.2,
    opener: Any = urllib.request.urlopen,
) -> ChatResult:
    """Call POST /chat/completions (DeepSeek OpenAI-compatible)."""
    url = f"{client.base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": client.model,
        "messages": messages,
    }
    # deepseek-reasoner rejects custom temperature
    if temperature is not None and "reasoner" not in client.model:
        payload["temperature"] = temperature

    headers = {"Content-Type": "application/json"}
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with opener(request, timeout=client.timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {body}") from exc

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


def chat_completion(
    client: ChatClient,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = 0.2,
    opener: Any = urllib.request.urlopen,
) -> str:
    return complete(client, messages, temperature=temperature, opener=opener).content
