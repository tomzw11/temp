"""Mock QwenPaw service for integration testing.

Runs a lightweight aiohttp server that mimics the QwenPaw
``POST /v1/chat/completions`` endpoint.  Used by unit tests so they
don't need a real QwenPaw deployment.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)

# Default ports used by tests — increment if tests run in parallel
DEFAULT_PORT = 18765


class MockQwenPawService:
    """A minimal aiohttp app that responds to /v1/chat/completions.

    Usage in tests::

        async with MockQwenPawService.run(port=18765) as url:
            client = QwenPawClient(QwenPawConfig(qwenpaw_api_url=url))
            result = await client.chat("hello")
            assert result["content"] == "mock response"
    """

    def __init__(
        self,
        *,
        response_content: str = "This is a mock QwenPaw response.",
        finish_reason: str = "stop",
        usage: dict[str, int] | None = None,
        status_code: int = 200,
        delay: float = 0.0,
        model_endpoint_received: asyncio.Event | None = None,
    ) -> None:
        self.response_content = response_content
        self.finish_reason = finish_reason
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        self.status_code = status_code
        self.delay = delay
        self.model_endpoint_received = model_endpoint_received
        self._last_request: dict[str, Any] = {}
        self._app = web.Application()
        self._app.router.add_post("/v1/chat/completions", self._handle_chat)
        self._runner: web.AppRunner | None = None

    # ------------------------------------------------------------------
    # Request handler
    # ------------------------------------------------------------------

    async def _handle_chat(self, request: web.Request) -> web.Response:
        """Handle POST /v1/chat/completions."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid json"}, status=400)

        self._last_request = body

        # Record that model_endpoint was received
        if self.model_endpoint_received is not None and "model_endpoint" in body:
            self.model_endpoint_received.set()

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.status_code != 200:
            return web.json_response(
                {"error": f"mock error {self.status_code}"},
                status=self.status_code,
            )

        response_body = {
            "id": "mock-qwenpaw-001",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "qwenpaw",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": self.response_content,
                    },
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": self.usage,
        }
        return web.json_response(response_body)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @classmethod
    async def run(cls, *, port: int = DEFAULT_PORT, **kwargs: Any) -> "MockQwenPawService":
        """Start the mock service and return the instance.

        Usage as async context manager::

            async with MockQwenPawService.run(port=18765) as svc:
                ...
        """
        svc = cls(**kwargs)
        svc._runner = web.AppRunner(svc._app)
        await svc._runner.setup()
        site = web.TCPSite(svc._runner, "localhost", port)
        await site.start()
        logger.info("MockQwenPawService started on http://localhost:%d", port)
        return svc

    async def stop(self) -> None:
        """Stop the mock service."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            logger.info("MockQwenPawService stopped")

    async def __aenter__(self) -> "MockQwenPawService":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    @property
    def url(self) -> str:
        return f"http://localhost:{DEFAULT_PORT}"

    @property
    def last_request(self) -> dict[str, Any]:
        """The JSON body of the most recent request."""
        return self._last_request