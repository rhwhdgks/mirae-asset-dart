"""Bounded, request-local timing events; never retain prompts or provider bodies."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import re
import time

_events = ContextVar("latency_diagnostic_events", default=None)
_STAGES = {"provider_http", "provider_limiter", "transport_attempt",
           "transport_backoff", "stage1_resolve", "stage1_plan"}


def record_latency_event(stage: str, **fields) -> None:
    events = _events.get()
    if events is None or len(events) >= 128 or stage not in _STAGES:
        return
    event = {"stage": stage}
    for key in ("elapsed_s", "attempt", "seed_overridden"):
        value = fields.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            event[key] = round(value, 6) if isinstance(value, float) else value
    outcome = fields.get("outcome")
    if isinstance(outcome, int) or (isinstance(outcome, str)
            and re.fullmatch(r"[A-Za-z0-9_]{1,80}", outcome)):
        event["outcome"] = outcome
    request_id = fields.get("request_id")
    if isinstance(request_id, str) and re.fullmatch(r"[0-9a-fA-F-]{36}", request_id):
        event["request_id"] = request_id
    events.append(event)


@contextmanager
def capture_latency():
    events = []
    token = _events.set(events)
    try:
        yield events
    finally:
        _events.reset(token)


@contextmanager
def latency_span(stage: str, **fields):
    started = time.perf_counter()
    outcome = "returned"
    try:
        yield
    except BaseException as exc:
        outcome = type(exc).__name__
        raise
    finally:
        record_latency_event(stage, elapsed_s=time.perf_counter() - started,
                             outcome=outcome, **fields)
