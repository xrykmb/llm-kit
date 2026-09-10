from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatClient:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout_sec: float = 60.0

    @classmethod
    def from_env(cls) -> ChatClient:
        return cls(
            api_key=os.environ.get("LLM_API_KEY", ""),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        )


def chat_completion(
    client: ChatClient,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    opener: Any = urllib.request.urlopen,
) -> str:
    """Call POST /chat/completions and return the first message content."""
    url = f"{client.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": client.model,
        "messages": messages,
        "temperature": temperature,
    }
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
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {data!r}") from exc
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected content type: {type(content)}")
    return content
