import asyncio

import review_queue


def test_retry_delay_is_exponential_and_bounded():
    assert review_queue.retry_delay_seconds(1, 5, 60) == 5
    assert review_queue.retry_delay_seconds(2, 5, 60) == 10
    assert review_queue.retry_delay_seconds(5, 5, 60) == 60
    assert review_queue.retry_delay_seconds(20, 5, 60) == 60


def test_worker_completes_successful_job(monkeypatch):
    completed = []

    async def scenario():
        stop = asyncio.Event()
        jobs = [{"id": 7, "attempts": 1, "max_attempts": 3}]

        async def claim(*_args, **_kwargs):
            return jobs.pop(0) if jobs else None

        async def complete(*_args, **kwargs):
            completed.append(kwargs["job_id"])
            stop.set()
            return True

        async def heartbeat(*_args, **_kwargs):
            return True

        async def handler(job):
            assert job["id"] == 7

        monkeypatch.setattr(review_queue, "claim_review_job", claim)
        monkeypatch.setattr(review_queue, "complete_review_job", complete)
        monkeypatch.setattr(review_queue, "heartbeat_review_job", heartbeat)

        await review_queue.review_worker_loop(
            object(),
            worker_id="worker-1",
            handler=handler,
            stop_event=stop,
            lease_seconds=30,
            poll_seconds=0.01,
            base_backoff_seconds=1,
            max_backoff_seconds=30,
        )

    asyncio.run(scenario())
    assert completed == [7]


def test_worker_schedules_retry_and_reports_failure(monkeypatch):
    failures = []

    async def scenario():
        stop = asyncio.Event()
        jobs = [{"id": 8, "attempts": 2, "max_attempts": 4}]

        async def claim(*_args, **_kwargs):
            return jobs.pop(0) if jobs else None

        async def fail(*_args, **_kwargs):
            return False, 10.0

        async def heartbeat(*_args, **_kwargs):
            return True

        async def handler(_job):
            raise RuntimeError("temporary outage")

        async def on_failure(job, error, terminal, delay):
            failures.append((job["id"], str(error), terminal, delay))
            stop.set()

        monkeypatch.setattr(review_queue, "claim_review_job", claim)
        monkeypatch.setattr(review_queue, "fail_review_job", fail)
        monkeypatch.setattr(review_queue, "heartbeat_review_job", heartbeat)

        await review_queue.review_worker_loop(
            object(),
            worker_id="worker-1",
            handler=handler,
            stop_event=stop,
            lease_seconds=30,
            poll_seconds=0.01,
            base_backoff_seconds=5,
            max_backoff_seconds=60,
            failure_handler=on_failure,
        )

    asyncio.run(scenario())
    assert failures == [(8, "temporary outage", False, 10.0)]
