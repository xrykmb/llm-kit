# CoT 与 DeepSeek

Chain-of-Thought（CoT）要求模型先写推理过程，再给结论，适合多步问题。

DeepSeek Chat（`deepseek-chat`）通过提示词做 CoT。

DeepSeek Reasoner（`deepseek-reasoner`）会在接口里返回 `reasoning_content` 字段，cli 可用 `--show-reasoning` 打印。

本项目默认 API 根路径是 `https://api.deepseek.com/v1`。
