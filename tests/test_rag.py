from pathlib import Path

from llm_kit.client import ChatClient, ChatResult
from llm_kit.cli import main
from llm_kit.rag.chunker import chunk_documents
from llm_kit.rag.loader import Document, load_paths
from llm_kit.rag.pipeline import ask, ingest
from llm_kit.rag.retriever import TfIdfRetriever


def test_chunk_and_retrieve_chinese(tmp_path: Path) -> None:
    docs = [
        Document(source="a.md", text="检索增强生成把找资料和写答案分开。"),
        Document(source="b.md", text="香蕉是一种水果，与语言模型无关。"),
    ]
    chunks = chunk_documents(docs, size=80, overlap=10)
    hits = TfIdfRetriever(chunks).search("RAG 怎么把检索和生成分开", k=1)
    assert hits
    assert "找资料" in hits[0].chunk.text


def test_ingest_and_ask(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "note.md").write_text("模块化 RAG 的步骤是加载、切块、检索、生成。", encoding="utf-8")
    index = tmp_path / "index.json"
    assert ingest([kb], index) >= 1

    def complete_fn(client, messages, **kwargs):
        assert "加载" in messages[-1]["content"] or "切块" in messages[-1]["content"]
        return ChatResult(content="思考：根据资料\n最终答案：四步", reasoning="内部推理")

    result = ask("RAG 有哪几步？", index, ChatClient(), cot=True, complete_fn=complete_fn)
    assert result.answer == "四步"
    assert result.hits


def test_cli_ingest_then_rag(tmp_path: Path, monkeypatch, capsys) -> None:
    kb = tmp_path / "doc.md"
    kb.write_text("DeepSeek 默认接口是 api.deepseek.com。", encoding="utf-8")
    index = tmp_path / "index.json"
    assert main(["ingest", str(kb), "--index", str(index)]) == 0

    monkeypatch.setattr(
        "llm_kit.cli.ask",
        lambda question, index_path, client, k=4, cot=True: type(
            "A",
            (),
            {
                "answer": "DeepSeek",
                "reasoning": "r",
                "raw": "最终答案：DeepSeek",
                "hits": (),
            },
        )(),
    )
    assert main(["rag", "用哪个 API？", "--index", str(index), "--json"]) == 0
    out = capsys.readouterr().out
    assert "DeepSeek" in out


def test_load_paths_skips_unknown(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x00")
    docs = load_paths([tmp_path])
    assert len(docs) == 1
