"""Retry utilities for QwenPaw HTTP calls.

Provides exponential-backoff retry decorators and a retry-aware client wrapper
for handling transient network failures, timeouts, and server errors.
"""

from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, Coroutine, TypeVar

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Coroutine[Any, Any, Any]])


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------


class RetryConfig:
    """Configuration for retry behaviour.

    Attributes:
        max_attempts: Maximum number of attempts (including the first call).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Maximum delay cap in seconds.
        backoff_multiplier: Exponential backoff multiplier.
        retryable_exceptions: Exception types (or status codes) that trigger a retry.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        retryable_exceptions: tuple[type[Exception], ...] = (
            asyncio.TimeoutError,
            ConnectionError,
            OSError,
        ),
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.retryable_exceptions = retryable_exceptions


# ---------------------------------------------------------------------------
# Retry decorator (async)
# ---------------------------------------------------------------------------


def async_retry(
    config: RetryConfig | None = None,
    *,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    backoff_multiplier: float | None = None,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
) -> Callable[[_F], _F]:
    """Decorator for async functions — retries on transient errors.

    Usage::

        @async_retry(max_attempts=3, base_delay=1.0)
        async def call_api(url: str) -> dict:
            ...

    :param config: RetryConfig instance (overrides keyword args).
    :param max_attempts: Max total attempts (default 3).
    :param base_delay: Initial delay in seconds (default 1.0).
    :param max_delay: Max delay cap (default 60.0).
    :param backoff_multiplier: Exponential multiplier (default 2.0).
    :param retryable_exceptions: Exceptions to retry on.
    """
    if config is not None:
        cfg = config
    else:
        cfg = RetryConfig(
            max_attempts=max_attempts or 3,
            base_delay=base_delay or 1.0,
            max_delay=max_delay or 60.0,
            backoff_multiplier=backoff_multiplier or 2.0,
            retryable_exceptions=retryable_exceptions or (
                asyncio.TimeoutError,
                ConnectionError,
                OSError,
            ),
        )

    def decorator(func: _F) -> _F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, cfg.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if not isinstance(exc, cfg.retryable_exceptions):
                        raise

                    last_exc = exc
                    if attempt == cfg.max_attempts:
                        logger.error(
                            "retry: all %d attempts failed for %s: %s",
                            cfg.max_attempts,
                            func.__name__,
                            exc,
                        )
                        raise

                    delay = min(
                        cfg.base_delay * (cfg.backoff_multiplier ** (attempt - 1)),
                        cfg.max_delay,
                    )
                    logger.warning(
                        "retry: attempt %d/%d for %s failed (%.1fs delay): %s",
                        attempt,
                        cfg.max_attempts,
                        func.__name__,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Retry runner (explicit loop, no decorator)
# ---------------------------------------------------------------------------


async def retry_call(
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    config: RetryConfig,
) -> Any:
    """Execute a coroutine factory with retry logic.

    Useful when you need programmatic retry without a decorator::

        result = await retry_call(
            lambda: client.chat("hello", model_endpoint=url),
            RetryConfig(max_attempts=3),
        )

    :param coro_factory: A callable that returns a fresh coroutine each time.
    :param config: Retry configuration.
    :returns: The coroutine result on success.
    :raises: The last exception after all attempts are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            if not isinstance(exc, config.retryable_exceptions):
                raise

            last_exc = exc
            if attempt == config.max_attempts:
                raise

            delay = min(
                config.base_delay * (config.backoff_multiplier ** (attempt - 1)),
                config.max_delay,
            )
            logger.warning(
                "retry_call: attempt %d/%d failed (%.1fs delay): %s",
                attempt,
                config.max_attempts,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]