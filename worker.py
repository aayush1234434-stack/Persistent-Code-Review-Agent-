"""Standalone durable review worker entry point."""

from __future__ import annotations

import asyncio
import contextlib
import signal

import main


async def run_worker() -> None:
    main.REVIEW_WORKER_ENABLED = True
    await main.startup()
    worker_task = main.app.state.review_worker_task
    stop_event = main.app.state.worker_stop
    loop = asyncio.get_running_loop()

    def request_shutdown() -> None:
        stop_event.set()
        worker_task.cancel()

    for signal_name in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signal_name, request_shutdown)

    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    finally:
        await main.shutdown()


if __name__ == "__main__":
    asyncio.run(run_worker())
