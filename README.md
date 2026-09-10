# llm-kit

模块化 **RAG + Chain-of-Thought**。生成默认走 **DeepSeek**；检索默认是稠密向量（余弦）+ BM25，再用 **RRF** 融合、**MMR** 去冗余。

DeepSeek 没有 embedding 接口。未配置向量服务时，用本地 hashing 向量（可离线跑通）；生产环境请接 OpenAI / SiliconFlow / Ollama 等兼容 `POST /embeddings` 的服务。

## 安装

```bash
python -m pip install -e ".[dev]"
```

## 配置

```powershell
$env:LLM_API_KEY="sk-..."
$env:LLM_BASE_URL="https://api.deepseek.com/v1"
$env:LLM_MODEL="deepseek-chat"
```

语义向量（推荐，例如 SiliconFlow 的 BGE）：

```powershell
$env:LLM_EMBED_API_KEY="sk-..."
$env:LLM_EMBED_BASE_URL="https://api.siliconflow.cn/v1"
$env:LLM_EMBED_MODEL="BAAI/bge-m3"
```

或 OpenAI：

```powershell
$env:LLM_EMBED_BASE_URL="https://api.openai.com/v1"
$env:LLM_EMBED_MODEL="text-embedding-3-small"
```

| 变量 | 默认 |
|------|------|
| `LLM_API_KEY` | 空 |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` |
| `LLM_MODEL` | `deepseek-chat` |
| `LLM_EMBED_BASE_URL` | 空（使用本地 hashing 向量） |
| `LLM_EMBED_MODEL` | `text-embedding-3-small` |
| `LLM_EMBED_API_KEY` | 回落到 `LLM_API_KEY` |

## 检索怎么排

1. 文本块做成 L2 归一化稠密向量，查询用 **余弦相似度**（归一化后即点积）。
2. 同时算 **Okapi BM25** 词法分。
3. 两路排序用 **Reciprocal Rank Fusion (RRF)** 合并。
4. 在候选集上做 **MMR**，降低近重复片段。

```bash
llm-kit ingest examples/kb
llm-kit rag "这个知识库里 RAG 怎么工作？" --show-reasoning
```

## 多轮对话 / 流式输出

默认流式打印 token。不带参数进入 REPL（历史会保留，可用 `/exit` `/reset` `/save`）。

```bash
llm-kit chat
llm-kit chat "用一句话解释 RAG" --show-reasoning
llm-kit chat "你好" --no-stream --session .llm-kit/chat.json
llm-kit rag
llm-kit rag "这个知识库里 RAG 怎么工作？" --show-reasoning
```

库用法：

```python
from llm_kit import ChatClient, Conversation, collect_stream

client = ChatClient.from_env()
convo = Conversation(system="你是助手", max_turns=8)
convo.add_user("1+1？")
result = collect_stream(client, convo.to_api_messages(), on_event=lambda e: print(e.text, end=""))
convo.add_assistant(result.content)
```

## 评测

```bash
llm-kit eval examples/eval.jsonl
```

## 作为库

```python
from pathlib import Path
from llm_kit import ChatClient, ingest, ask

index = Path(".llm-kit/index.json")
ingest([Path("examples/kb")], index)
answer = ask("RAG 分哪几步？", index, ChatClient.from_env(), cot=True)
print(answer.answer)
```

## 开发

```bash
python -m pytest
```

## License

MIT
