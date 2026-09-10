import io
import json
from pathlib import Path
from urllib.error import HTTPError

from llm_kit.client import ChatClient, chat_completion
from llm_kit.eval import evaluate_jsonl
from llm_kit.cli import main


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        return None


def test_chat_completion_parses_message() -> None:
    def opener(request, timeout):
        body = json.loads(request.data.decode("utf-8"))
        assert body["model"] == "demo"
        assert request.get_header("Authorization") == "Bearer sk-test"
        return FakeResponse({"choices": [{"message": {"content": "hello"}}]})

    client = ChatClient(api_key="sk-test", base_url="http://local/v1", model="demo")
    assert chat_completion(client, [{"role": "user", "content": "hi"}], opener=opener) == "hello"


def test_chat_completion_http_error() -> None:
    def opener(request, timeout):
        raise HTTPError("http://local/v1/chat/completions", 401, "nope", hdrs=None, fp=io.BytesIO(b'{"error":"no"}'))

    client = ChatClient(base_url="http://local/v1")
    try:
        chat_completion(client, [{"role": "user", "content": "hi"}], opener=opener)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "401" in str(exc)


def test_evaluate_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "a", "prompt": "p", "expect_contains": ["yes"]}\n',
        encoding="utf-8",
    )
    client = ChatClient()

    def complete(client, messages):
        return "the answer is yes"

    results = evaluate_jsonl(path, client, complete=complete)
    assert results[0].passed is True
    assert results[0].missing == ()


def test_cli_eval_exit_code(tmp_path: Path, monkeypatch, capsys) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"prompt": "p", "expect_contains": ["nope"]}\n', encoding="utf-8")

    def complete(client, messages, **kwargs):
        return "something else"

    monkeypatch.setattr("llm_kit.cli.evaluate_jsonl", lambda p, c: evaluate_jsonl(p, c, complete=complete))
    assert main(["eval", str(path)]) == 1
    out = capsys.readouterr().out
    assert "0/1 passed" in out
