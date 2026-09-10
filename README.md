# llm-kit

模块化 **RAG + Chain-of-Thought**，默认对接 **DeepSeek** 的 OpenAI 兼容接口。

检索层是可替换的 TF-IDF（无向量库依赖）；生成层走 DeepSeek Chat，可用 CoT 提示，也兼容 `deepseek-reasoner` 的 `reasoning_content`。

## 安装

```bash
python -m pip install -e ".[dev]"
```

## 配置（DeepSeek）

到 [DeepSeek 开放平台](https://platform.deepseek.com/) 申请 API Key。

```powershell
$env:LLM_API_KEY="sk-..."
$env:LLM_BASE_URL="https://api.deepseek.com/v1"
$env:LLM_MODEL="deepseek-chat"
```

更强推理可改成：

```powershell
$env:LLM_MODEL="deepseek-reasoner"
```

| 变量 | 默认 |
|------|------|
| `LLM_API_KEY` | 空 |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` |
| `LLM_MODEL` | `deepseek-chat` |

## RAG 流程

`load` → `chunk` → `TfIdfRetriever` → `CoT prompt` → `DeepSeek`

```bash
llm-kit ingest examples/kb
llm-kit rag "这个知识库里 RAG 怎么工作？" --show-reasoning
```

关闭 CoT：

```bash
llm-kit rag "问题" --no-cot
```

## 聊天

```bash
llm-kit chat "用一句话解释 RAG"
llm-kit chat --show-reasoning "逐步说明检索为什么能减少幻觉"
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

替换检索器：实现与 `TfIdfRetriever` 相同的 `add` / `search`，再接到 `rag.pipeline`。

## 开发

```bash
python -m pytest
```

## License

MIT
