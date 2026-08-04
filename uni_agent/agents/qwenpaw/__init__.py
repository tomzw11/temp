"""QwenPaw: remote agent integration via OpenAI-compatible API (RFC #5790 AgentGateway pattern).

Primary interface for training:
    ``uni_agent.agents.qwenpaw.agent.run_qwenpaw`` — task_runner function
    that receives ``SessionHandle`` directly from the framework.

Standalone client for testing:
    ``uni_agent.agents.qwenpaw.QwenPawClient`` — async HTTP client.

Architecture:
    uni-agent communicates with a deployed QwenPaw service via its
    OpenAI-compatible ``POST /v1/chat/completions`` endpoint.
    uni-agent does NOT import AgentScope or manage model clients.
    It forwards the Gateway session URL to QwenPaw via the ``model_endpoint``
    field, so QwenPaw routes all LLM calls through the Gateway for trajectory capture.

QwenPaw-side requirement:
    QwenPaw's /v1/chat/completions endpoint must accept an optional ``model_endpoint``
    field. When provided, QwenPaw uses this URL as its LLM backend.

Dependencies:
    aiohttp (for async HTTP communication with QwenPaw service)
"""

from .agent import QwenPawClient, QwenPawConfig, run_qwenpaw
from .multi_turn import MultiTurnQwenPawRunner
from .retry import RetryConfig, async_retry, retry_call
from .reward import CompositeReward, GSM8KReward, RewardFunction
from .streaming import collect_stream_response, sse_chat_stream

__all__ = [
    "QwenPawClient",
    "QwenPawConfig",
    "run_qwenpaw",
    "MultiTurnQwenPawRunner",
    "RetryConfig",
    "async_retry",
    "retry_call",
    "CompositeReward",
    "GSM8KReward",
    "RewardFunction",
    "collect_stream_response",
    "sse_chat_stream",
]
