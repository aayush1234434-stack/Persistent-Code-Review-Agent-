"""Durable PostgreSQL queue primitives for pull-request review jobs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from observability import record_review_job_event


logger = logging.getLogger("pr_review.queue")

JobHandler = Callable[[dict[str, Any]], Awaitable[None]]
FailureHandler = Callable[[dict[str, Any], Exception, bool, float], Awaitable[None]]


def retry_delay_seconds(attempts: int, base_seconds: float, max_seconds: float) -> float:
    """Return bounded exponential backoff for a one-based attempt count."""
    exponent = max(int(attempts) - 1, 0)
    return min(float(max_seconds), float(base_seconds) * (2**exponent))


async def run_migrations(pool, migrations_dir: Path) -> None:
    """Apply each SQL migration exactly once under a transaction."""
    migration_paths = sorted(migrations_dir.glob("*.sql"))
    async with pool.acquire() as connection:
        await connection.execute(
            "SELECT pg_advisory_lock(hashtextextended('pr-review-migrations', 0))"
        )
        try:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            for path in migration_paths:
                async with connection.transaction():
                    applied = await connection.fetchval(
                        "SELECT 1 FROM schema_migrations WHERE version = $1",
                        path.name,
                    )
                    if applied:
                        continue
                    await connection.execute(path.read_text(encoding="utf-8"))
                    await connection.execute(
                        "INSERT INTO schema_migrations (version) VALUES ($1)",
                        path.name,
                    )
        finally:
            try:
                await connection.execute(
                    "SELECT pg_advisory_unlock(hashtextextended('pr-review-migrations', 0))"
                )
            except Exception:
                logger.exception("Unable to release migration advisory lock")


async def enqueue_review_job(
    pool,
    *,
    idempotency_key: str,
    metadata: dict[str, Any],
    max_attempts: int,
) -> tuple[int, bool]:
    """Persist an event and return ``(job_id, created)``."""
    row = await pool.fetchrow(
        """
        INSERT INTO review_jobs (
            idempotency_key,
            repo,
            pr_number,
            source_sha,
            event_action,
            payload,
            max_attempts
        )
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
        ON CONFLICT (idempotency_key) DO NOTHING
        RETURNING id
        """,
        idempotency_key,
        metadata["repository"],
        metadata["pr_number"],
        metadata.get("source_sha"),
        metadata.get("action", "unknown"),
        json.dumps(metadata),
        max_attempts,
    )
    if row is not None:
        record_review_job_event("enqueue", "queued", job_id=row["id"])
        return int(row["id"]), True

    existing_id = await pool.fetchval(
        "SELECT id FROM review_jobs WHERE idempotency_key = $1",
        idempotency_key,
    )
    if existing_id is None:
        raise RuntimeError("Unable to resolve idempotent review job")
    record_review_job_event("enqueue", "duplicate", job_id=existing_id)
    return int(existing_id), False


async def claim_review_job(pool, *, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
    """Atomically claim one available job, including jobs with expired leases."""
    row = await pool.fetchrow(
        """
        WITH candidate AS (
            SELECT id
            FROM review_jobs
            WHERE (
                    status IN ('queued', 'retry')
                    AND available_at <= NOW()
                  )
               OR (
                    status = 'running'
                    AND locked_at < NOW() - make_interval(secs => $2)
                  )
            ORDER BY available_at, id
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        UPDATE review_jobs AS job
        SET status = 'running',
            attempts = job.attempts + 1,
            locked_at = NOW(),
            locked_by = $1,
            started_at = COALESCE(job.started_at, NOW()),
            updated_at = NOW()
        FROM candidate
        WHERE job.id = candidate.id
        RETURNING job.*
        """,
        worker_id,
        lease_seconds,
    )
    if row is None:
        return None
    claimed = dict(row)
    record_review_job_event(
        "claim",
        "running",
        job_id=claimed.get("id"),
        attempt=claimed.get("attempts"),
    )
    return claimed


async def attach_review_to_job(pool, job_id: int, review_id: int) -> None:
    await pool.execute(
        """
        UPDATE review_jobs
        SET review_id = $2, updated_at = NOW()
        WHERE id = $1
        """,
        job_id,
        review_id,
    )


async def heartbeat_review_job(pool, *, job_id: int, worker_id: str) -> bool:
    result = await pool.execute(
        """
        UPDATE review_jobs
        SET locked_at = NOW(), updated_at = NOW()
        WHERE id = $1 AND status = 'running' AND locked_by = $2
        """,
        job_id,
        worker_id,
    )
    return result == "UPDATE 1"


async def complete_review_job(pool, *, job_id: int, worker_id: str) -> bool:
    result = await pool.execute(
        """
        UPDATE review_jobs
        SET status = 'completed',
            locked_at = NULL,
            locked_by = NULL,
            completed_at = NOW(),
            updated_at = NOW()
        WHERE id = $1 AND status = 'running' AND locked_by = $2
        """,
        job_id,
        worker_id,
    )
    completed = result == "UPDATE 1"
    if completed:
        record_review_job_event("finish", "completed", job_id=job_id)
    return completed


async def fail_review_job(
    pool,
    *,
    job: dict[str, Any],
    worker_id: str,
    error: Exception,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
) -> tuple[bool, float]:
    attempts = int(job.get("attempts", 1))
    max_attempts = int(job.get("max_attempts", 1))
    terminal = attempts >= max_attempts
    delay = 0.0 if terminal else retry_delay_seconds(
        attempts,
        base_backoff_seconds,
        max_backoff_seconds,
    )
    status = "failed" if terminal else "retry"
    await pool.execute(
        """
        UPDATE review_jobs
        SET status = $3,
            available_at = CASE
                WHEN $3 = 'retry' THEN NOW() + make_interval(secs => $4)
                ELSE available_at
            END,
            locked_at = NULL,
            locked_by = NULL,
            last_error = $5,
            completed_at = CASE WHEN $3 = 'failed' THEN NOW() ELSE NULL END,
            updated_at = NOW()
        WHERE id = $1 AND status = 'running' AND locked_by = $2
        """,
        int(job["id"]),
        worker_id,
        status,
        delay,
        f"{type(error).__name__}: {error}"[:2000],
    )
    record_review_job_event(
        "finish",
        status,
        job_id=job.get("id"),
        attempt=attempts,
        retry_in_seconds=delay,
        error_type=type(error).__name__,
    )
    return terminal, delay


async def _heartbeat_loop(
    pool,
    *,
    job_id: int,
    worker_id: str,
    interval_seconds: float,
) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        renewed = await heartbeat_review_job(pool, job_id=job_id, worker_id=worker_id)
        if not renewed:
            return


async def wait_for_work(stop_event: asyncio.Event, poll_seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
    except asyncio.TimeoutError:
        return


async def review_worker_loop(
    pool,
    *,
    worker_id: str,
    handler: JobHandler,
    stop_event: asyncio.Event,
    lease_seconds: int,
    poll_seconds: float,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
    failure_handler: FailureHandler | None = None,
) -> None:
    """Continuously claim and process durable jobs until shutdown."""
    heartbeat_interval = max(min(lease_seconds / 3, 30), 1)
    while not stop_event.is_set():
        try:
            job = await claim_review_job(pool, worker_id=worker_id, lease_seconds=lease_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unable to claim a review job")
            await wait_for_work(stop_event, poll_seconds)
            continue
        if job is None:
            await wait_for_work(stop_event, poll_seconds)
            continue

        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(
                pool,
                job_id=int(job["id"]),
                worker_id=worker_id,
                interval_seconds=heartbeat_interval,
            )
        )
        try:
            await handler(job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            try:
                terminal, delay = await fail_review_job(
                    pool,
                    job=job,
                    worker_id=worker_id,
                    error=exc,
                    base_backoff_seconds=base_backoff_seconds,
                    max_backoff_seconds=max_backoff_seconds,
                )
            except Exception:
                logger.exception("Unable to record review job failure", extra={"job_id": job.get("id")})
                continue
            if failure_handler is not None:
                try:
                    await failure_handler(job, exc, terminal, delay)
                except Exception:
                    logger.exception("Review job failure callback failed", extra={"job_id": job.get("id")})
            logger.exception(
                "Review job failed",
                extra={
                    "job_id": job.get("id"),
                    "status": "failed" if terminal else "retry",
                    "duration_ms": delay * 1000,
                },
            )
        else:
            try:
                await complete_review_job(pool, job_id=int(job["id"]), worker_id=worker_id)
            except Exception:
                logger.exception("Unable to complete review job", extra={"job_id": job.get("id")})
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
