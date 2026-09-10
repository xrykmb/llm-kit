import json

from llm_kit.client import ChatClient, chat_completion, complete
from llm_kit.cot import build_rag_messages, extract_final_answer


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        return None


def test_parses_deepseek_reasoning() -> None:
    def opener(request, timeout):
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "最终答案：42",
                            "reasoning_content": "一步步算",
                        }
                    }
                ]
            }
        )

    client = ChatClient(model="deepseek-reasoner")
    result = complete(client, [{"role": "user", "content": "q"}], opener=opener)
    assert result.content == "最终答案：42"
    assert result.reasoning == "一步步算"
    assert chat_completion(client, [{"role": "user", "content": "q"}], opener=opener) == "最终答案：42"


def test_reasoner_omits_temperature() -> None:
    captured = {}

    def opener(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    complete(
        ChatClient(model="deepseek-reasoner"),
        [{"role": "user", "content": "q"}],
        opener=opener,
    )
    assert "temperature" not in captured["body"]


def test_cot_messages_and_final_answer() -> None:
    messages = build_rag_messages("RAG 分几步？", ["加载、切块、检索、生成"], cot=True)
    assert messages[0]["role"] == "system"
    assert "Chain-of-Thought" in messages[0]["content"]
    assert "资料" in messages[1]["content"]
    assert extract_final_answer("思考：先检索\n最终答案：四步") == "四步"
