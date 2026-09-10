import io

from llm_kit.client import ChatClient, collect_stream, complete_stream
from llm_kit.session import Conversation
from llm_kit.stream import events_from_payload, iter_sse_payloads


class FakeStream:
    def __init__(self, text: str) -> None:
        self._buf = io.BytesIO(text.encode("utf-8"))

    def readline(self) -> bytes:
        return self._buf.readline()

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *args) -> None:
        return None


def test_iter_sse_payloads_stops_at_done() -> None:
    lines = [
        "event: ignore",
        'data: {"choices":[{"delta":{"content":"你"}}]}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"x"}}]}',
    ]
    payloads = list(iter_sse_payloads(lines))
    assert len(payloads) == 1
    events = events_from_payload(payloads[0])
    assert events[0].text == "你"


def test_collect_stream_joins_reasoning_and_content() -> None:
    sse = (
        'data: {"choices":[{"delta":{"reasoning_content":"想"}}]}\n'
        'data: {"choices":[{"delta":{"content":"答"}}]}\n'
        'data: {"choices":[{"delta":{"content":"案"}}]}\n'
        "data: [DONE]\n"
    )

    def opener(request, timeout):
        body = __import__("json").loads(request.data.decode("utf-8"))
        assert body["stream"] is True
        return FakeStream(sse)

    seen: list[str] = []
    result = collect_stream(
        ChatClient(model="demo"),
        [{"role": "user", "content": "q"}],
        opener=opener,
        on_event=lambda event: seen.append(event.kind),
    )
    assert result.reasoning == "想"
    assert result.content == "答案"
    assert seen == ["reasoning", "content", "content"]
    assert list(complete_stream(ChatClient(model="demo"), [{"role": "user", "content": "q"}], opener=opener))


def test_conversation_trim_and_roundtrip(tmp_path) -> None:
    convo = Conversation(system="sys", max_turns=2)
    convo.add_user("u1")
    convo.add_assistant("a1")
    convo.add_user("u2")
    convo.add_assistant("a2")
    convo.add_user("u3")
    convo.add_assistant("a3")
    assert [m["content"] for m in convo.messages] == ["u2", "a2", "u3", "a3"]
    api = convo.to_api_messages()
    assert api[0] == {"role": "system", "content": "sys"}
    path = tmp_path / "s.json"
    convo.save(path)
    loaded = Conversation.load(path)
    assert loaded.messages == convo.messages
