from __future__ import annotations

COT_SYSTEM = """你是严谨的助手。先逐步推理（Chain-of-Thought），再给出结论。
要求：
1. 只用提供的资料回答；资料不足时明确说不知道。
2. 推理过程写在「思考：」之后。
3. 最终结论单独写在「最终答案：」之后，且结论必须能被资料支撑。
"""

FINAL_MARKER = "最终答案："


def build_rag_messages(
    question: str,
    contexts: list[str],
    *,
    cot: bool = True,
) -> list[dict[str, str]]:
    numbered = "\n\n".join(f"[{i + 1}]\n{text}" for i, text in enumerate(contexts)) or "（无检索结果）"
    if cot:
        user = (
            f"资料：\n{numbered}\n\n问题：{question}\n\n"
            "请先逐步思考，再输出最终答案。"
        )
        return [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": user},
        ]

    return [
        {
            "role": "system",
            "content": "你是助手。只根据资料作答，不要编造。资料不足时说不知道。",
        },
        {"role": "user", "content": f"资料：\n{numbered}\n\n问题：{question}"},
    ]


def extract_final_answer(text: str) -> str:
    if FINAL_MARKER in text:
        return text.split(FINAL_MARKER, 1)[1].strip()
    return text.strip()
