"""Destructive integration check against the dedicated Docker Compose test database."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import asyncpg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from review_queue import claim_review_job, complete_review_job, enqueue_review_job, run_migrations

DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://pr_user:pr_pass@127.0.0.1:5432/pr_review",
)


async def wait_for_pool(timeout_seconds: float = 45) -> asyncpg.Pool:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(1)
    raise RuntimeError(f"PostgreSQL did not recover within {timeout_seconds}s: {last_error}")


async def scenario() -> None:
    unique = uuid.uuid4().hex
    metadata = {
        "repository": "integration/recovery",
        "pr_number": int(unique[:6], 16),
        "source_sha": unique,
        "action": "synchronize",
    }
    pool = await wait_for_pool()
    await run_migrations(pool, ROOT / "migrations")
    job_id, created = await enqueue_review_job(
        pool,
        idempotency_key=f"restart-recovery-{unique}",
        metadata=metadata,
        max_attempts=3,
    )
    assert created
    claimed = await claim_review_job(pool, worker_id="worker-before-restart", lease_seconds=1)
    assert claimed and int(claimed["id"]) == job_id and int(claimed["attempts"]) == 1
    await pool.close()

    subprocess.run(
        ["docker", "compose", "restart", "db"],
        cwd=ROOT,
        check=True,
    )
    pool = await wait_for_pool()
    await asyncio.sleep(1.2)
    recovered = await claim_review_job(pool, worker_id="worker-after-restart", lease_seconds=1)
    assert recovered and int(recovered["id"]) == job_id
    assert int(recovered["attempts"]) == 2
    assert await complete_review_job(pool, job_id=job_id, worker_id="worker-after-restart")
    row = await pool.fetchrow("SELECT status, attempts FROM review_jobs WHERE id = $1", job_id)
    assert row and row["status"] == "completed" and int(row["attempts"]) == 2
    await pool.execute("DELETE FROM review_jobs WHERE id = $1", job_id)
    await pool.close()
    print("PASS: queued job survived PostgreSQL restart and was reclaimed after lease expiry")


if __name__ == "__main__":
    asyncio.run(scenario())
