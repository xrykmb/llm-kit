from llm_kit.rag.embedder import OpenAICompatEmbedder, cosine, l2_normalize
from llm_kit.rag.similarity import maximal_marginal_relevance, reciprocal_rank_fusion


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = __import__("json").dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        return None


def test_rrf_prefers_overlap() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60)
    assert fused["a"] > fused["b"]
    assert fused["c"] > fused["b"]


def test_mmr_diversifies() -> None:
    query = l2_normalize([1.0, 0.0])
    vectors = {
        "near": l2_normalize([0.99, 0.1]),
        "near_dup": l2_normalize([0.98, 0.12]),
        "other": l2_normalize([0.2, 0.98]),
    }
    chosen = maximal_marginal_relevance(
        query,
        ["near", "near_dup", "other"],
        vectors,
        k=2,
        lambda_=0.2,
    )
    assert chosen[0] == "near"
    assert "other" in chosen


def test_openai_compat_embedder_normalizes() -> None:
    def opener(request, timeout):
        return FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 3.0]},
                    {"index": 0, "embedding": [4.0, 0.0]},
                ]
            }
        )

    embedder = OpenAICompatEmbedder("k", "http://local/v1", "m", opener=opener)
    vectors = embedder.embed_many(["a", "b"])
    assert cosine(vectors[0], [1.0, 0.0]) > 0.99
    assert cosine(vectors[1], [0.0, 1.0]) > 0.99
