"""Tests for retry utilities."""

from __future__ import annotations

import asyncio

import pytest

from uni_agent.agents.qwenpaw.retry import RetryConfig, async_retry, retry_call


class TestRetryDecorator:
    """Tests for the @async_retry decorator."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """No retry needed when the function succeeds immediately."""
        call_count = 0

        @async_retry(max_attempts=3, base_delay=0.01)
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeed()
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        """Retries on retryable exceptions and eventually succeeds."""
        call_count = 0

        @async_retry(max_attempts=3, base_delay=0.01)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        result = await flaky()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        """Raises the last exception after all attempts fail."""
        call_count = 0

        @async_retry(max_attempts=2, base_delay=0.01)
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("always down")

        with pytest.raises(ConnectionError, match="always down"):
            await always_fail()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Non-retryable exceptions propagate immediately."""
        call_count = 0

        @async_retry(max_attempts=3, base_delay=0.01)
        async def type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError, match="not retryable"):
            await type_error()

        assert call_count == 1


class TestRetryCall:
    """Tests for the retry_call utility."""

    @pytest.mark.asyncio
    async def test_success(self):
        """retry_call succeeds on the first try."""
        result = await retry_call(
            lambda: asyncio.sleep(0.001, result="ok"),
            RetryConfig(max_attempts=3, base_delay=0.01),
        )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retry_and_succeed(self):
        """retry_call retries and succeeds."""
        call_count = 0

        def factory():
            nonlocal call_count

            async def inner():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ConnectionError("fail")
                return "ok"

            return inner()

        result = await retry_call(
            factory,
            RetryConfig(max_attempts=3, base_delay=0.01),
        )
        assert result == "ok"
        assert call_count == 2