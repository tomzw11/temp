## Summary

Add QwenPaw integration as a **task_runner** (RFC #5790 AgentGateway pattern). uni-agent communicates with a deployed QwenPaw service via its OpenAI-compatible `POST /v1/chat/completions` endpoint. uni-agent does **not** import AgentScope or manage model clients — it only forwards the Gateway session URL so QwenPaw routes all LLM calls through the Gateway for trajectory capture.

## Motivation

[QwenPaw](https://github.com/agentscope-ai/QwenPaw) is a personal AI assistant built on AgentScope's `ReActAgent`. It is a **deployed remote service** (FastAPI + Console + Channels), not a CLI tool like Claude Code. It cannot be launched per rollout task.

This PR implements the **task_runner** pattern: the framework owns the session lifecycle (create → run → finalize), and `run_qwenpaw` is a callback that receives `SessionHandle` directly. It calls QwenPaw's `POST /v1/chat/completions` with `model_endpoint: session.base_url`, so QwenPaw uses the Gateway as its LLM backend.

## Architecture

### Data Flow (per training step)

```
verl 训练 loop（第 N 步）
  │
  ├─ 1. Actor 更新权重 → sync_to_vllm() → vLLM 权重已更新
  │
  ├─ 2. 创建 Gateway session → SessionHandle(base_url="http://gateway:port/v1")
  │
  ├─ 3. run_qwenpaw(session=SessionHandle, raw_prompt=...)
  │     │
  │     │  POST /v1/chat/completions
  │     │  {model, messages, model_endpoint: session.base_url}
  │     │  ──────────────────────────────────────> QwenPaw Service（远程）
  │     │                                                │
  │     │                                         ReAct loop:
  │     │                                           tool_call → LLM → tool_call → LLM → ...
  │     │                                                │            │
  │     │                                    每次 LLM 调用走 model_endpoint:
  │     │                              POST {base_url}/v1/chat/completions
  │     │                              ──────> Gateway ──路由──> vLLM（verl 管理）
  │     │                              <────── OpenAI response
  │     │                                       │
  │     │                              Gateway 透明捕获轨迹:
  │     │                                prompt_ids, response_ids, logprobs, response_mask
  │     │
  │     │  <── OpenAI JSON response ──────────
  │     │
  │     └─ return TaskResult(finished=..., extra_info=...)
  │
  ├─ 4. finalize_session() → list[Trajectory]
  │
  ├─ 5. TransferQueue → verl PPO Trainer
  │
  └─ 6. 计算 advantage → 更新 Actor 权重 → sync_to_vllm()
       │
       ▼
  第 N+1 步：vLLM 权重已更新，新 session 自动使用新权重
```

### Key Design Decisions

**1. task_runner pattern (RFC #5790)**

The framework owns the session lifecycle. `run_qwenpaw` is a callback that receives `SessionHandle` directly. No Agent class, no registry entry, no sandbox.

**2. Remote service, not embedded kernel**

uni-agent does NOT import AgentScope. It communicates with a deployed QwenPaw service via HTTP. QwenPaw manages its own model clients internally.

**3. Gateway session URL forwarded as model_endpoint**

`session.base_url` is passed to QwenPaw as `model_endpoint`. QwenPaw uses this URL for all internal LLM calls. Gateway transparently captures token-level trajectories.

**4. vLLM remains under verl control**

QwenPaw does not own or manage vLLM. The Gateway is a proxy that routes to verl's vLLM instance. Weight sync happens automatically via verl's `sync_rollout_weights()` between training steps.

**5. OpenAI-compatible endpoint**

Uses standard `POST /v1/chat/completions` (not QwenPaw's custom `/api/console/chat`). Request/response follow OpenAI format — no SSE parsing needed.

### QwenPaw-side requirement

QwenPaw's `/v1/chat/completions` endpoint must accept an optional `model_endpoint` field. When provided, QwenPaw uses this URL as its LLM backend instead of its default.

## Usage

### YAML config

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

### Standalone test

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

## Files changed

| File | Change |
|------|--------|
| `uni_agent/agents/qwenpaw/agent.py` | `QwenPawClient` + `run_qwenpaw` + `QwenPawConfig` (唯一实现) |
| `uni_agent/agents/qwenpaw/__init__.py` | 导出 `QwenPawClient`, `QwenPawConfig`, `run_qwenpaw` |
| `uni_agent/agents/qwenpaw/agent.md` | 架构文档 |
| `uni_agent/agents/registry.py` | 移除 qwenpaw 条目（不再是 Agent，是 runner） |

---

## AI 图片生成提示词

### 中文版

```
一张平面风格的架构流程图，白色背景，配色简洁只用蓝、灰、黑三色，所有文字使用大号加粗字体，字号至少 24pt。

画的整体结构：两个大虚线框分别代表两个系统，左侧是"verl"，右侧是"uni-agent"。大框外右侧放一个独立的远程服务模块。

=== 左侧大框：verl（蓝色虚线框）===

框内从上到下排列三个子模块，用粗箭头连接：

1. "Actor（模型）"（蓝色圆角矩形）
2. "vLLM Server（推理引擎）"（灰色圆角矩形）
3. "PPO Trainer（训练循环）"（蓝色圆角矩形）

内部箭头：
- Actor → vLLM Server：箭头标注"sync_rollout_weights（每轮开始前）"
- PPO Trainer → Actor：右侧弧形返回箭头，标注"⑨ 更新权重 → 下一轮"

=== 右侧大框：uni-agent（灰色虚线框）===

框内从上到下排列三个子模块：

1. "Gateway"（蓝色圆角矩形，标注"FastAPI + 轨迹捕获"）
2. "Framework"（灰色圆角矩形，标注"Session 生命周期管理"）
3. "QwenPaw Runner"（灰色圆角矩形，标注"task_runner"）

内部箭头：
- Framework ↔ Gateway：双向箭头，标注"① 创建 Session / ⑦ 收集轨迹"
- Framework → QwenPaw Runner：箭头标注"② dispatch"

=== 右侧独立模块：QwenPaw Service ===

一个独立的蓝色圆角矩形，放在 uni-agent 大框右侧，标注"远程部署"。

模块内部用虚线框标注"ReAct Loop: tool_call ↔ LLM"。

=== 跨系统连线（按步骤编号）===

步骤 ③：QwenPaw Runner → QwenPaw Service
  箭头标注"POST /v1/chat/completions\n{model_endpoint}"

步骤 ④：QwenPaw Service → Gateway
  箭头标注"LLM 调用\n/v1/chat/completions"

步骤 ⑤：Gateway → vLLM Server
  箭头标注"路由推理"

步骤 ⑥ 返回：vLLM Server → Gateway → QwenPaw Service
  返回箭头标注"OpenAI Response"

步骤 ⑦：Gateway → Framework
  箭头标注"Trajectory\n(prompt_ids, logprobs)"

步骤 ⑧：Framework → PPO Trainer
  箭头标注"TransferQueue"

=== 底部反馈回路 ===

PPO Trainer 右侧引出一条弧形箭头，从底部绕回到 Actor，标注"⑨ 更新权重 → sync_rollout_weights → 下一轮迭代"。

=== 样式要求 ===

- 扁平化设计，无阴影无渐变
- 线条粗 2px，箭头清晰
- 模块间距均匀，整体干净
- 步骤编号用白底黑字的圆形图标标注在箭头旁
- 大虚线框的标题用大号粗体字标注在框的左上角
```

### English Version

```
A flat-style architecture flow diagram, white background, simple color scheme using only blue, gray, and black. All text in large bold font, minimum 24pt.

Two large dashed border boxes representing two systems: "verl" on the left, "uni-agent" on the right. An independent remote service module placed outside to the right.

=== Left box: verl (blue dashed border) ===

Three sub-modules arranged top-to-bottom, connected by thick arrows:

1. "Actor" (blue rounded rectangle)
2. "vLLM Server" (gray rounded rectangle, subtitle "Inference Engine")
3. "PPO Trainer" (blue rounded rectangle, subtitle "Training Loop")

Internal arrows:
- Actor → vLLM Server: label "sync_rollout_weights (before each round)"
- PPO Trainer → Actor: curved return arrow on the right side, label "⑨ Update Weights → Next Round"

=== Right box: uni-agent (gray dashed border) ===

Three sub-modules arranged top-to-bottom:

1. "Gateway" (blue rounded rectangle, subtitle "FastAPI + Trajectory Capture")
2. "Framework" (gray rounded rectangle, subtitle "Session Lifecycle")
3. "QwenPaw Runner" (gray rounded rectangle, subtitle "task_runner")

Internal arrows:
- Framework ↔ Gateway: bidirectional arrow, label "① Create Session / ⑦ Collect Trajectory"
- Framework → QwenPaw Runner: arrow label "② dispatch"

=== Independent module: QwenPaw Service ===

A standalone blue rounded rectangle placed to the right of the uni-agent box, labeled "Remote".

Inside: dashed box labeled "ReAct Loop: tool_call ↔ LLM".

=== Cross-system connections (numbered) ===

Step ③: QwenPaw Runner → QwenPaw Service
  Label: "POST /v1/chat/completions\n{model_endpoint}"

Step ④: QwenPaw Service → Gateway
  Label: "LLM Calls\n/v1/chat/completions"

Step ⑤: Gateway → vLLM Server
  Label: "Route Inference"

Step ⑤ return: vLLM Server → Gateway → QwenPaw Service
  Return arrow label: "OpenAI Response"

Step ⑦: Gateway → Framework
  Label: "Trajectory\n(prompt_ids, logprobs)"

Step ⑧: Framework → PPO Trainer
  Label: "TransferQueue"

=== Bottom feedback loop ===

A curved arrow from the right side of PPO Trainer, looping along the bottom back to Actor, labeled "⑨ Update Weights → sync_rollout_weights → Next Iteration".

=== Style ===

- Flat design, no shadows, no gradients
- 2px thick lines, clear arrowheads
- Even spacing between modules, clean layout
- Step numbers in white-on-black circle badges next to arrows
- Large dashed border box titles in bold at the top-left corner of each box
```