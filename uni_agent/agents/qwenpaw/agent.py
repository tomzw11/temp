"""QwenPaw integration: HTTP client + task_runner for a deployed QwenPaw service.

uni-agent does NOT import AgentScope or manage model clients. It communicates
with QwenPaw via its OpenAI-compatible ``POST /v1/chat/completions`` endpoint
and forwards the Gateway session URL so QwenPaw routes all LLM calls through
the Gateway for trajectory capture.

Architecture (RFC #5790 AgentGateway pattern):

    Framework                            QwenPaw Service (remote)
    ─────────                            ────────────────────────
    create_session() → SessionHandle
      │
      │  run_qwenpaw(session=handle)
      │    │
      │    │  POST /v1/chat/completions
      │    │  {model, messages, model_endpoint: session.base_url}
      │    │ ──────────────────────────────>  ReAct loop
      │    │                                       │
      │    │                              POST {base_url}/v1/chat/completions
      │    │                              ──────> Gateway → tokenize → vLLM → record
      │    │                              <────── OpenAI response
      │    │
      │    │  <── OpenAI JSON response ────────
      │    │
      │    │  POST session.reward_info_url (if report_reward=True)
      │
    finalize_session() → list[Trajectory]

QwenPaw-side requirement:
    QwenPaw's /v1/chat/completions endpoint MUST accept an optional
    ``model_endpoint`` field in the request body. When provided, QwenPaw
    uses this URL as the base URL for all internal LLM calls.

YAML configuration::

    agent_runners:
      qwenpaw:
        runner_fqn: uni_agent.agents.qwenpaw.agent.run_qwenpaw
        runner_kwargs:
          qwenpaw_api_url: "http://10.0.0.5:8088"
          agent_id: "default"
          auth_token: null
          request_timeout: 600.0
          report_reward: false
        dispatch_mode: inline_async
        max_concurrent_sessions: 1
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

import aiohttp
from pydantic import BaseModel, Field

from uni_agent.tasks.base import TaskResult

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class QwenPawConfig(BaseModel):
    """Configuration for connecting to a remote QwenPaw service."""

    qwenpaw_api_url: str = Field(
        default="http://localhost:8088",
        description="Base URL of the deployed QwenPaw service.",
    )
    agent_id: str = Field(
        default="default",
        description="QwenPaw agent ID (X-Agent-Id header).",
    )
    auth_token: str | None = Field(
        default=None,
        description="Bearer token for remote QwenPaw access.",
    )
    request_timeout: float = Field(
        default=600.0,
        description="Total timeout (seconds) for a single QwenPaw API call.",
    )
    connect_timeout: float = Field(
        default=30.0,
        description="Connection timeout (seconds).",
    )


# ---------------------------------------------------------------------------
# HTTP Client
# ---------------------------------------------------------------------------


class QwenPawClient:
    """Async HTTP client for QwenPaw's OpenAI-compatible /v1/chat/completions.

    Usage::

        client = QwenPawClient(QwenPawConfig(qwenpaw_api_url="http://10.0.0.5:8088"))
        result = await client.chat("What is 2+2?", model_endpoint="http://gateway:8000/v1")
        print(result["content"])
    """

    def __init__(self, config: QwenPawConfig) -> None:
        self._cfg = config

    async def chat(
        self,
        user_message: str,
        *,
        model_endpoint: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a chat request and return the parsed result.

        Args:
            user_message: The user's message text.
            model_endpoint: If set, forwarded to QwenPaw so it uses this URL
                as its LLM backend instead of its default. This is how the
                Gateway session URL is injected for trajectory capture.
            session_id: Optional session ID for QwenPaw's session management.

        Returns:
            Dict with keys: ``content``, ``finish_reason``, ``usage`` (optional),
            ``error`` (optional), ``session_id``.
        """
        if session_id is None:
            session_id = f"uniagent-{uuid.uuid4().hex[:12]}"

        payload: dict[str, Any] = {
            "model": "qwenpaw",
            "messages": [{"role": "user", "content": user_message}],
            "stream": False,
        }
        if model_endpoint is not None:
            payload["model_endpoint"] = model_endpoint

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-Agent-Id": self._cfg.agent_id,
        }
        if self._cfg.auth_token:
            headers["Authorization"] = f"Bearer {self._cfg.auth_token}"

        url = f"{self._cfg.qwenpaw_api_url.rstrip('/')}/v1/chat/completions"

        timeout = aiohttp.ClientTimeout(
            total=self._cfg.request_timeout,
            connect=self._cfg.connect_timeout,
        )

        try:
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        error_body = await resp.text()
                        raise RuntimeError(
                            f"QwenPaw API returned {resp.status}: {error_body[:500]}"
                        )

                    data = await resp.json()

            choice = data.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason", "stop")
            message = choice.get("message", {})
            content = message.get("content", "")
            usage = data.get("usage")

            result: dict[str, Any] = {
                "session_id": session_id,
                "content": content,
                "finish_reason": finish_reason,
            }
            if usage is not None:
                result["usage"] = usage

            return result

        except aiohttp.ClientError as exc:
            logger.exception("QwenPaw HTTP request failed: %s", exc)
            return {
                "session_id": session_id,
                "content": "",
                "finish_reason": "error",
                "error": f"HTTP error: {exc}",
            }
        except Exception as exc:
            logger.exception("QwenPaw client failed: %s", exc)
            return {
                "session_id": session_id,
                "content": "",
                "finish_reason": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }


# ---------------------------------------------------------------------------
# Task Runner (RFC #5790 AgentRunner contract)
# ---------------------------------------------------------------------------


async def run_qwenpaw(
    *,
    session: SessionHandle,
    raw_prompt: Any = None,
    sample_index: int | None = None,
    tools_kwargs: dict[str, Any] | None = None,
    # --- QwenPaw connection (from YAML runner_kwargs) ---
    qwenpaw_api_url: str = "http://localhost:8088",
    agent_id: str = "default",
    auth_token: str | None = None,
    request_timeout: float = 600.0,
    connect_timeout: float = 30.0,
    report_reward: bool = False,
    **_: Any,
) -> TaskResult:
    """Run one QwenPaw episode against a Gateway session.

    Satisfies the framework's ``AgentRunner`` contract. The framework injects
    ``session`` (SessionHandle) directly; all other kwargs come from YAML
    ``runner_kwargs``.

    Args:
        session: Gateway session handle. ``session.base_url`` is forwarded to
            QwenPaw as ``model_endpoint``.
        raw_prompt: The task prompt (string, message list, or dict).
        sample_index: Sample index for logging.
        tools_kwargs: Additional task metadata (unused).
        qwenpaw_api_url: Base URL of the deployed QwenPaw service.
        agent_id: QwenPaw agent ID (X-Agent-Id header).
        auth_token: Bearer token for remote QwenPaw access.
        request_timeout: Total timeout (seconds).
        connect_timeout: Connection timeout (seconds).
        report_reward: If True, POST reward info to ``session.reward_info_url``.

    Returns:
        TaskResult with finished status and extra info.
    """
    if session.base_url is None:
        raise ValueError(
            "qwenpaw runner: session.base_url is None. "
            "The Gateway session must provide a base URL."
        )

    user_message = _extract_user_message(raw_prompt)

    client = QwenPawClient(
        QwenPawConfig(
            qwenpaw_api_url=qwenpaw_api_url,
            agent_id=agent_id,
            auth_token=auth_token,
            request_timeout=request_timeout,
            connect_timeout=connect_timeout,
        )
    )

    logger.info(
        "qwenpaw runner start: sample_index=%s qwenpaw_url=%s",
        sample_index,
        qwenpaw_api_url,
    )

    result = await client.chat(
        user_message,
        model_endpoint=session.base_url,
    )

    finished = result.get("finish_reason") not in (None, "error")
    content = result.get("content", "")
    error_info = result.get("error")

    extra_info: dict[str, Any] = {
        "qwenpaw_session_id": result.get("session_id"),
        "agent_id": agent_id,
        "qwenpaw_api_url": qwenpaw_api_url,
        "content": content,
    }
    if result.get("usage"):
        extra_info["usage"] = result["usage"]
    if error_info:
        extra_info["error"] = error_info

    # Post reward to Gateway session (best-effort)
    reward_posted = False
    if report_reward and session.reward_info_url:
        reward_posted = await _post_reward_info(
            session.reward_info_url,
            reward=1.0 if finished else 0.0,
            finished=finished,
            extra_info=extra_info,
        )

    logger.info(
        "qwenpaw runner done: sample_index=%s finished=%s reward_posted=%s",
        sample_index,
        finished,
        reward_posted,
    )

    return TaskResult(
        reward=1.0 if finished else 0.0,
        finished=finished,
        extra_info=extra_info,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_user_message(raw_prompt: Any) -> str:
    """Extract a plain-text user message from the raw prompt."""
    if isinstance(raw_prompt, str):
        return raw_prompt

    if isinstance(raw_prompt, list):
        parts: list[str] = []
        for msg in raw_prompt:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("system", "user") and content:
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
        if parts:
            return "\n\n".join(parts)

    if isinstance(raw_prompt, dict):
        text = raw_prompt.get("prompt") or raw_prompt.get("text") or raw_prompt.get("content")
        if isinstance(text, str):
            return text

    raise ValueError(
        f"qwenpaw runner: cannot extract user message from raw_prompt type "
        f"{type(raw_prompt).__name__}"
    )


async def _post_reward_info(
    reward_info_url: str,
    reward: float,
    finished: bool,
    extra_info: dict[str, Any] | None = None,
) -> bool:
    """Best-effort POST of reward metadata to the Gateway session."""
    reward_info: dict[str, Any] = {"reward": reward, "finished": finished}
    if extra_info:
        reward_info["extra_info"] = extra_info

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async with http.post(
                reward_info_url, json={"reward_info": reward_info}
            ) as response:
                response.raise_for_status()
        logger.debug("posted reward_info to %s", reward_info_url)
        return True
    except Exception as exc:
        logger.warning(
            "failed to post reward_info to %s: %s: %s",
            reward_info_url,
            type(exc).__name__,
            exc,
        )
        return False
