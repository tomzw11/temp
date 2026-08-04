"""Streaming (SSE) client for QwenPaw's /v1/chat/completions endpoint.

When ``stream=True`` is set, QwenPaw returns Server-Sent Events (SSE) chunks
of the form ``data: <json>\n\n``.  This module provides an async generator
that yields each chunk as it arrives, plus a helper that collects the full
response.

Usage::

    async for chunk in sse_chat_stream("http://...", payload, headers):
        print(chunk["choices"][0]["delta"].get("content", ""))
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, overload

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE chunk parser
# ---------------------------------------------------------------------------


async def _parse_sse_stream(
    response: aiohttp.ClientResponse,
) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed SSE chunks from an aiohttp response body.

    Each SSE event is expected to be a line starting with ``data: `` followed
    by a JSON payload. The stream ends when a ``[DONE]`` marker is received.
    """
    async for line in response.content:
        line_str = line.decode("utf-8").strip()
        if not line_str:
            continue
        if not line_str.startswith("data: "):
            continue

        payload = line_str[6:]  # Strip "data: " prefix
        if payload == "[DONE]":
            return

        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("sse: failed to parse chunk: %s", payload[:200])
            continue


# ---------------------------------------------------------------------------
# Streaming chat generator
# ---------------------------------------------------------------------------


async def sse_chat_stream(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: aiohttp.ClientTimeout | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """POST to the chat completions endpoint and yield SSE chunks.

    :param url: Full URL of the completions endpoint.
    :param payload: Request body (must include ``"stream": True``).
    :param headers: HTTP headers.
    :param timeout: aiohttp ClientTimeout (defaults to 300s total, 30s connect).
    :yields: Parsed SSE chunk dicts.
    :raises RuntimeError: If the HTTP status is not 200.
    """
    payload.setdefault("stream", True)

    if timeout is None:
        timeout = aiohttp.ClientTimeout(total=300.0, connect=30.0)

    async with aiohttp.ClientSession(timeout=timeout) as http:
        async with http.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                raise RuntimeError(
                    f"QwenPaw streaming API returned {resp.status}: {error_body[:500]}"
                )
            async for chunk in _parse_sse_stream(resp):
                yield chunk


# ---------------------------------------------------------------------------
# Collect full response from stream
# ---------------------------------------------------------------------------


async def collect_stream_response(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: aiohttp.ClientTimeout | None = None,
) -> dict[str, Any]:
    """Collect all SSE chunks into a single complete response dict.

    Returns a dict with keys: ``content``, ``finish_reason``, ``usage``,
    ``stream_chunks`` (list of all raw chunks).

    :returns: Aggregated response dict.
    """
    full_content: list[str] = []
    finish_reason = "stop"
    usage: dict[str, Any] | None = None
    chunks: list[dict[str, Any]] = []

    async for chunk in sse_chat_stream(url, payload, headers, timeout=timeout):
        chunks.append(chunk)

        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            text = delta.get("content", "")
            if text:
                full_content.append(text)
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = fr

        if "usage" in chunk:
            usage = chunk["usage"]

    return {
        "content": "".join(full_content),
        "finish_reason": finish_reason,
        "usage": usage,
        "stream_chunks": chunks,
    }