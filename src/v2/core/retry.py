"""
retry.py — Exponential Backoff Retry Decorator (v2)

Provides a universal retry decorator for any function that may
fail transiently (LLM API calls, web requests, Qdrant operations).

Design choices:
  - Uses a plain decorator so it works on sync and async functions.
  - Exponential backoff with jitter to avoid thundering herd.
  - Configurable exception types so it only retries on retriable errors.
  - Logs every retry attempt at WARNING level so it's observable.

Usage:
    @with_retry(max_attempts=3, base_delay=1.0)
    def call_llm(prompt):
        return llm.invoke(prompt)

    @with_retry(exceptions=(httpx.RequestError,), max_attempts=5)
    async def fetch_url(url):
        ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from typing import Callable, Type, Tuple

logger = logging.getLogger(__name__)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator factory: retry the wrapped function on failure.

    Args:
        max_attempts:   Total number of attempts (including the first).
        base_delay:     Initial sleep duration in seconds.
        max_delay:      Cap on sleep duration.
        backoff_factor: Multiplier applied to delay on each retry.
        jitter:         Add up to 30% random jitter to avoid stampede.
        exceptions:     Only retry on these exception types.

    Returns:
        A decorator that wraps sync or async functions.
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                delay = base_delay
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as exc:
                        if attempt == max_attempts:
                            logger.error(
                                f"{func.__name__}: all {max_attempts} attempts failed. "
                                f"Last error: {exc}"
                            )
                            raise
                        sleep = min(delay, max_delay)
                        if jitter:
                            sleep *= 1 + random.uniform(0, 0.3)
                        logger.warning(
                            f"{func.__name__}: attempt {attempt}/{max_attempts} failed "
                            f"({exc}), retrying in {sleep:.1f}s…"
                        )
                        await asyncio.sleep(sleep)
                        delay *= backoff_factor
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                delay = base_delay
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as exc:
                        if attempt == max_attempts:
                            logger.error(
                                f"{func.__name__}: all {max_attempts} attempts failed. "
                                f"Last error: {exc}"
                            )
                            raise
                        sleep = min(delay, max_delay)
                        if jitter:
                            sleep *= 1 + random.uniform(0, 0.3)
                        logger.warning(
                            f"{func.__name__}: attempt {attempt}/{max_attempts} failed "
                            f"({exc}), retrying in {sleep:.1f}s…"
                        )
                        time.sleep(sleep)
                        delay *= backoff_factor
            return sync_wrapper
    return decorator
