"""Request-local cooperative deadline shared by the single canonical worker.

Python cannot safely stop a running thread.  The server therefore keeps its
one-worker semaphore until the worker really returns, but long pure-Python
backend loops must have a cheap way to notice that the HTTP execution budget
has already expired and return themselves.  This module deliberately carries
only a monotonic deadline; it has no HTTP, canonical-data, or provider state.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import time
from typing import Iterator


class RequestDeadlineExceeded(TimeoutError):
    """A cooperative backend boundary observed the request execution deadline."""


_deadline_monotonic: ContextVar[float | None] = ContextVar(
    "dart_request_deadline_monotonic", default=None)


@contextmanager
def request_deadline(deadline_monotonic: float | None) -> Iterator[None]:
    """Install one finite absolute deadline for work on the canonical thread."""
    token = _deadline_monotonic.set(deadline_monotonic)
    try:
        yield
    finally:
        _deadline_monotonic.reset(token)


def current_deadline_monotonic() -> float | None:
    """Return the current worker deadline, if this is a budgeted request."""
    return _deadline_monotonic.get()


def ensure_request_time_remaining() -> float | None:
    """Return remaining seconds or abort a cooperative long-running loop.

    A ``None`` result means the caller was invoked outside a server request
    (offline tests/builds), so it must retain its historical unbounded mode.
    """
    deadline = _deadline_monotonic.get()
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RequestDeadlineExceeded("request execution deadline exceeded")
    return remaining


__all__ = [
    "RequestDeadlineExceeded",
    "current_deadline_monotonic",
    "ensure_request_time_remaining",
    "request_deadline",
]
