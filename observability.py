"""Structured logging and Prometheus metrics for the PR review service."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

logger = logging.getLogger("pr_review")

HTTP_REQUESTS = Counter(
    "pr_review_http_requests_total",
    "HTTP requests by method, path, and status",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "pr_review_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
REVIEWS_TOTAL = Counter(
    "pr_review_reviews_total",
    "PR reviews by terminal outcome",
    ["outcome"],
)
LLM_CALLS = Counter(
    "pr_review_llm_calls_total",
    "LLM invocations by node and model",
    ["node", "model", "status"],
)
LLM_LATENCY = Histogram(
    "pr_review_llm_call_duration_seconds",
    "LLM call latency in seconds",
    ["node", "model"],
    buckets=(0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)
LLM_TOKENS = Counter(
    "pr_review_llm_tokens_total",
    "LLM tokens consumed by type",
    ["token_type"],
)
LLM_COST = Counter(
    "pr_review_llm_estimated_cost_usd_total",
    "Estimated LLM spend in USD",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "event",
            "repo",
            "pr_number",
            "review_id",
            "delivery_id",
            "source_sha",
            "node",
            "model",
            "status",
            "duration_ms",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "estimated_cost_usd",
            "method",
            "path",
            "status_code",
            "error_type",
        ):
            if hasattr(record, key):
                value = getattr(record, key)
                if value is not None:
                    payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging() -> None:
    root = logging.getLogger()
    if any(isinstance(handler.formatter, JsonFormatter) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def log_event(level: int, event: str, message: str, **fields: Any) -> None:
    logger.log(level, message, extra={"event": event, **fields})


def record_review_outcome(outcome: str, **fields: Any) -> None:
    REVIEWS_TOTAL.labels(outcome=outcome).inc()
    log_event(logging.INFO, "review_outcome", f"Review {outcome}", status=outcome, **fields)


def record_llm_call(
    *,
    node: str,
    model: str,
    status: str,
    duration_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    **fields: Any,
) -> None:
    LLM_CALLS.labels(node=node, model=model, status=status).inc()
    LLM_LATENCY.labels(node=node, model=model).observe(duration_ms / 1000.0)
    if prompt_tokens:
        LLM_TOKENS.labels(token_type="prompt").inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(token_type="completion").inc(completion_tokens)
    if total_tokens:
        LLM_TOKENS.labels(token_type="total").inc(total_tokens)
    if estimated_cost_usd:
        LLM_COST.inc(estimated_cost_usd)
    log_event(
        logging.INFO if status == "success" else logging.WARNING,
        "llm_call",
        f"LLM call {status} for {node}",
        node=node,
        model=model,
        status=status,
        duration_ms=round(duration_ms, 2),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=round(estimated_cost_usd, 6),
        **fields,
    )


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def register_request_logging_middleware(app) -> None:
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - started
            path = request.url.path
            method = request.method
            HTTP_REQUESTS.labels(method=method, path=path, status=str(status_code)).inc()
            HTTP_LATENCY.labels(method=method, path=path).observe(duration)
            log_event(
                logging.INFO,
                "http_request",
                f"{method} {path} {status_code}",
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=round(duration * 1000, 2),
            )
