"""Tests for QwenPawClient (HTTP client)."""

from __future__ import annotations

import pytest

from uni_agent.agents.qwenpaw.agent import QwenPawClient, QwenPawConfig

from .mock_service import DEFAULT_PORT, MockQwenPawService


class TestQwenPawClient:
    """Unit tests for the HTTP client."""

    @pytest.mark.asyncio
    async def test_chat_success(self):
        """Client returns correct content on 200 OK."""
        async with MockQwenPawService.run(
            port=DEFAULT_PORT,
            response_content="Hello from mock QwenPaw!",
        ) as svc:
            client = QwenPawClient(
                QwenPawConfig(qwenpaw_api_url=svc.url)
            )
            result = await client.chat("What is 2+2?")

        assert result["content"] == "Hello from mock QwenPaw!"
        assert result["finish_reason"] == "stop"
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    @pytest.mark.asyncio
    async def test_chat_forwards_model_endpoint(self):
        """Client includes model_endpoint in the request payload."""
        async with MockQwenPawService.run(port=DEFAULT_PORT) as svc:
            client = QwenPawClient(
                QwenPawConfig(qwenpaw_api_url=svc.url)
            )
            await client.chat(
                "hello",
                model_endpoint="http://gateway:8000/v1",
            )

        assert svc.last_request.get("model_endpoint") == "http://gateway:8000/v1"

    @pytest.mark.asyncio
    async def test_chat_error_status(self):
        """Client returns error info on non-200 status."""
        async with MockQwenPawService.run(
            port=DEFAULT_PORT,
            status_code=500,
        ) as svc:
            client = QwenPawClient(
                QwenPawConfig(qwenpaw_api_url=svc.url)
            )
            result = await client.chat("hello")

        assert result["finish_reason"] == "error"
        assert "500" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_chat_custom_session_id(self):
        """Client accepts a custom session_id."""
        async with MockQwenPawService.run(port=DEFAULT_PORT) as svc:
            client = QwenPawClient(
                QwenPawConfig(qwenpaw_api_url=svc.url)
            )
            result = await client.chat("hello", session_id="my-custom-id")

        assert result["session_id"] == "my-custom-id"

    @pytest.mark.asyncio
    async def test_chat_connection_refused(self):
        """Client handles connection errors gracefully."""
        client = QwenPawClient(
            QwenPawConfig(
                qwenpaw_api_url="http://127.0.0.1:19999",  # no service here
                connect_timeout=0.5,
            )
        )
        result = await client.chat("hello")

        assert result["finish_reason"] == "error"
        assert result["error"] is not None