from __future__ import annotations

import re

_TOKEN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    words = _TOKEN.findall(lowered)
    compact = re.sub(r"\s+", "", lowered)
    grams = [compact[i : i + 2] for i in range(max(0, len(compact) - 1))]
    return words + grams
