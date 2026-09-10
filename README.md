# llm-kit

面向大模型应用的小型工具包：用 **OpenAI 兼容** HTTP 接口发聊天请求，并在 JSONL 上做简单评测。

适用于 OpenAI、DeepSeek、通义兼容模式、本地 vLLM / Ollama（OpenAI 接口）等。

## 安装

```bash
python -m pip install -e ".[dev]"
```

Python 3.10+，无第三方运行时依赖。

## 配置

环境变量：

| 变量 | 含义 | 默认 |
|------|------|------|
| `LLM_API_KEY` | API Key | 空（部分本地服务可不填） |
| `LLM_BASE_URL` | 接口根路径 | `https://api.openai.com/v1` |
| `LLM_MODEL` | 模型名 | `gpt-4o-mini` |

## 聊天

```bash
llm-kit chat "用一句话解释什么是 RAG"
llm-kit chat --system "你是严谨的助手" "列出 3 条评测指标"
```

## 评测

`examples/eval.jsonl` 每行一个用例：

```json
{"id": "rag-1", "prompt": "RAG 的全称是什么？", "expect_contains": ["Retrieval"]}
```

```bash
llm-kit eval examples/eval.jsonl
llm-kit eval examples/eval.jsonl --json
```

评测只做 **子串包含** 检查，用来快速回归提示词或网关，不是学术基准。

## 作为库使用

```python
from llm_kit import ChatClient, chat_completion, evaluate_jsonl

client = ChatClient()
text = chat_completion(client, messages=[{"role": "user", "content": "hi"}])
```

## 开发

```bash
python -m pytest
```

## License

MIT
