# QwenPaw 集成详解（task_runner 模式 + v1/chat/completions）

## 架构概览

基于 RFC #5790 AgentGateway 模式，uni-agent 通过 QwenPaw 的 OpenAI 兼容 `POST /v1/chat/completions` 端点与已部署的 QwenPaw 服务通信。uni-agent **不导入 AgentScope、不创建模型客户端**，只传递 Gateway session URL。

```
Framework._run_session_with_concurrency_limit()
  │
  ├─ gateway_manager.create_session()
  │     → SessionHandle(base_url="http://gateway:port/v1", reward_info_url=...)
  │
  ├─ run_qwenpaw(session=SessionHandle, raw_prompt=..., ...)
  │     │
  │     │  POST /v1/chat/completions
  │     │  {model, messages, model_endpoint: session.base_url}
  │     │ ──────────────────────────────> QwenPaw Service (remote)
  │     │                                       │
  │     │                              POST {base_url}/v1/chat/completions
  │     │                              ──────> Gateway → tokenize → vLLM → record
  │     │                              <────── OpenAI response
  │     │
  │     │  <── OpenAI JSON response ────────
  │     │
  │     │  POST session.reward_info_url (if report_reward=True)
  │     │
  │     └─ return TaskResult(finished=..., extra_info=...)
  │
  ├─ gateway_manager.finalize_session()
  │     → list[Trajectory]
  │
  └─ _write_session_trajectories_to_tq()
```

---

## 文件结构

| 文件 | 角色 |
|------|------|
| `uni_agent/agents/qwenpaw/agent.py` | **唯一实现**：`QwenPawClient` + `run_qwenpaw` + `QwenPawConfig` |
| `uni_agent/agents/qwenpaw/__init__.py` | 导出 `QwenPawClient`, `QwenPawConfig`, `run_qwenpaw` |

YAML 直接引用 `uni_agent.agents.qwenpaw.agent.run_qwenpaw`，无需额外的 re-export 文件。

---

## 逐行详解：agent.py

### QwenPawConfig（第 73-95 行）

```python
class QwenPawConfig(BaseModel):
    qwenpaw_api_url: str = "http://localhost:8088"
    agent_id: str = "default"
    auth_token: str | None = None
    request_timeout: float = 600.0
    connect_timeout: float = 30.0
```

纯 pydantic `BaseModel`，不继承 `AgentConfig`。

### QwenPawClient（第 103-203 行）

```python
class QwenPawClient:
    async def chat(self, user_message: str, *, model_endpoint: str | None = None) -> dict:
```

核心方法，发送 OpenAI 兼容请求：

```python
payload = {
    "model": "qwenpaw",
    "messages": [{"role": "user", "content": user_message}],
    "stream": False,
}
if model_endpoint is not None:
    payload["model_endpoint"] = model_endpoint  # ← Gateway session URL

url = f"{qwenpaw_api_url}/v1/chat/completions"
```

响应解析（标准 OpenAI JSON）：

```python
data = await resp.json()
choice = data["choices"][0]
content = choice["message"]["content"]
finish_reason = choice["finish_reason"]
usage = data.get("usage")
```

返回 dict：`{"session_id", "content", "finish_reason", "usage"?, "error"?}`

### run_qwenpaw（第 211-313 行）

```python
async def run_qwenpaw(
    *,
    session: SessionHandle,          # ← 框架注入
    raw_prompt: Any = None,
    qwenpaw_api_url: str = "...",    # ← YAML runner_kwargs
    ...
) -> TaskResult:
```

内部调用 `QwenPawClient`：

```python
client = QwenPawClient(QwenPawConfig(...))
result = await client.chat(user_message, model_endpoint=session.base_url)
```

然后构建 `TaskResult`，可选回传 reward 到 `session.reward_info_url`。

### _extract_user_message（第 321-351 行）

从 `raw_prompt` 提取纯文本。支持字符串、OpenAI messages 列表、dict。

### _post_reward_info（第 354-381 行）

Best-effort POST reward 到 Gateway session。

---

## 与旧版对比

| | 旧版（/api/console/chat + SSE） | 新版（v1/chat/completions） |
|---|---|---|
| 端点 | `POST /api/console/chat` | `POST /v1/chat/completions` |
| 请求格式 | `{input, session_id, channel}` | `{model, messages}` |
| 响应格式 | SSE 流 `data: {status, output}` | JSON `{choices: [{message}]}` |
| 解析复杂度 | 逐行解析 SSE | 一次 `resp.json()` |
| 代码重复 | agent.py 和 runner 各自实现 | agent.py 唯一实现，runner re-export |

---

## YAML 配置示例

```yaml
agent_runners:
  qwenpaw:
    runner_fqn: uni_agent.agents.qwenpaw.agent.run_qwenpaw
    runner_kwargs:
      qwenpaw_api_url: "http://10.0.0.5:8088"
      agent_id: "default"
      auth_token: null
      request_timeout: 600.0
      connect_timeout: 30.0
      report_reward: false
    dispatch_mode: inline_async
    max_concurrent_sessions: 1
```

---

## QwenPaw 侧需要的改动

QwenPaw 的 `/v1/chat/completions` 需要支持 `model_endpoint` 字段：

```python
# 在 QwenPaw 的 chat completions handler 中
model_endpoint = body.get("model_endpoint")
if model_endpoint:
    # AgentScope 的 OpenAIChatModel 读取 OPENAI_BASE_URL
    import os
    os.environ["OPENAI_BASE_URL"] = model_endpoint
```

---

## 如果 QwenPaw 不能适配 model_endpoint

| 方案 | 说明 | 轨迹捕获 |
|------|------|----------|
| 反向代理 | QwenPaw 侧部署代理，按请求头路由到 Gateway | 完整 |
| Whitebox + task_runner | uni-agent import AgentScope 内核，传 `session.base_url` | 完整 |
| 黑箱模式 | 只拿 QwenPaw 最终输出，不算 token 级轨迹 | 无 |

---

## 独立测试示例

```python
import asyncio
from uni_agent.agents.qwenpaw import QwenPawClient, QwenPawConfig

async def test():
    client = QwenPawClient(QwenPawConfig(
        qwenpaw_api_url="http://10.0.0.5:8088",
        agent_id="default",
    ))
    result = await client.chat("What is 2+2?")
    print(result["content"])

asyncio.run(test())
```

---

## Prefix KV-Cache（吞吐优化）

### 原理

vLLM 内置 `--enable-prefix-caching`，基于 token hash 自动识别共享前缀并复用 KV-cache。QwenPaw 每个 turn 的请求天然包含相同前缀（system prompt + task + tools），vLLM 自动处理。

### QwenPaw 侧

**零改动**。无需特殊 header 或 cache ID。

### Gateway 侧需要的改动

唯一缺失的是 **session 路由亲和性**：同一 session 的所有请求必须路由到同一 vLLM worker。

| 改动点 | 文件 | 内容 |
|--------|------|------|
| `SessionHandle` 加 `worker_id` | `types.py` | 记录分配的 worker |
| `create_session()` 分配 worker | `gateway.py` | 创建 session 时固定 worker |
| 路由层按 session 保持亲和性 | `gateway.py` | 从 URL 提取 session_id，路由到固定 worker |
| `sync_rollout_weights()` 清空 | `gateway.py` | 权重更新后 `_session_workers.clear()` |
| vLLM 启动参数 | 部署配置 | `--enable-prefix-caching` |

详见 `prefix_cache.py`。

### 预期收益

| 前缀占比 | 轮数 | 加速比 |
|---------|------|--------|
| 80%+ | 5+ | 4–5× |
| 50% | 3–4 | 2–3× |
| 30% | 2 | 1.5× |
