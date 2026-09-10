from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Conversation:
    """Multi-turn chat history. System prompt is kept; older turns are trimmed."""

    system: str = ""
    max_turns: int = 12
    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def reset(self) -> None:
        self.messages.clear()

    def to_api_messages(self, extra_user: str | None = None) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        out.extend(self.messages)
        if extra_user:
            out.append({"role": "user", "content": extra_user})
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"system": self.system, "max_turns": self.max_turns, "messages": self.messages},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Conversation:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            system=str(data.get("system") or ""),
            max_turns=int(data.get("max_turns") or 12),
            messages=list(data.get("messages") or []),
        )

    def _trim(self) -> None:
        max_msgs = max(self.max_turns, 1) * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]
