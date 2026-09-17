import json
from datetime import datetime, timezone
from typing import Any


class FakeRecord(dict):
    """Minimal asyncpg.Record stand-in for dict-like row access."""

    def __getitem__(self, key):
        return super().__getitem__(key)


class FakePool:
    def __init__(self):
        self.reviews: dict[int, dict[str, Any]] = {}
        self.jobs: dict[int, dict[str, Any]] = {}
        self._next_id = 1
        self._next_job_id = 1

    def acquire(self):
        return FakeContext(self)

    def transaction(self):
        return FakeContext(self)

    async def fetchval(self, query: str, *args):
        if "SELECT id FROM review_jobs WHERE idempotency_key" in query:
            for job in self.jobs.values():
                if job["idempotency_key"] == args[0]:
                    return job["id"]
        if "SELECT 1" in query:
            return 1
        if "MAX(review_version)" in query:
            repo, pr_number = args
            versions = [
                row["review_version"]
                for row in self.reviews.values()
                if row["repo"] == repo and row["pr_number"] == pr_number
            ]
            return max(versions, default=0) + 1
        return None

    async def fetchrow(self, query: str, *args):
        if "INSERT INTO review_jobs" in query:
            for job in self.jobs.values():
                if job["idempotency_key"] == args[0]:
                    return None
            job_id = self._next_job_id
            self._next_job_id += 1
            payload = json.loads(args[5]) if isinstance(args[5], str) else args[5]
            self.jobs[job_id] = {
                "id": job_id,
                "idempotency_key": args[0],
                "repo": args[1],
                "pr_number": args[2],
                "source_sha": args[3],
                "event_action": args[4],
                "payload": payload,
                "status": "queued",
                "attempts": 0,
                "max_attempts": args[6],
                "review_id": None,
            }
            return FakeRecord({"id": job_id})
        if "INSERT INTO pr_reviews" in query:
            review_id = self._next_id
            self._next_id += 1
            pr_context = json.loads(args[3]) if isinstance(args[3], str) else args[3]
            now = datetime.now(timezone.utc)
            row = {
                "id": review_id,
                "repo": args[0],
                "pr_number": args[1],
                "status": args[2],
                "pr_context": pr_context,
                "result": None,
                "source_sha": args[4],
                "review_version": args[5],
                "webhook_delivery_id": args[6],
                "lock_version": 0,
                "created_at": now,
                "updated_at": now,
            }
            self.reviews[review_id] = row
            return FakeRecord(row)
        if "UPDATE pr_reviews" in query and "status = ANY" in query:
            review_id, new_status, expected = args
            row = self.reviews.get(review_id)
            if row is None or row["status"] not in expected:
                return None
            row["status"] = new_status
            row["lock_version"] += 1
            row["updated_at"] = datetime.now(timezone.utc)
            return FakeRecord(row)
        if "UPDATE pr_reviews" in query and "result = $3::jsonb" in query:
            review_id, new_status, result_json, expected = args
            row = self.reviews.get(review_id)
            if row is None or row["status"] != expected:
                return None
            row["status"] = new_status
            row["result"] = json.loads(result_json) if isinstance(result_json, str) else result_json
            row["lock_version"] += 1
            row["updated_at"] = datetime.now(timezone.utc)
            return FakeRecord(row)
        if "review_version < $3" in query:
            repo, pr_number, review_version = args
            candidates = [
                row for row in self.reviews.values()
                if row["repo"] == repo
                and row["pr_number"] == pr_number
                and row["review_version"] < review_version
                and row.get("result") is not None
            ]
            if not candidates:
                return None
            return FakeRecord(max(candidates, key=lambda row: row["review_version"]))
        if "FROM pr_reviews" in query and "WHERE id = $1" in query:
            row = self.reviews.get(args[0])
            return FakeRecord(row) if row else None
        if "source_sha" in query:
            repo, pr_number, source_sha = args
            for row in self.reviews.values():
                ctx = row.get("pr_context") or {}
                if (
                    row["repo"] == repo
                    and row["pr_number"] == pr_number
                    and row.get("source_sha", ctx.get("source_sha")) == source_sha
                ):
                    return FakeRecord(row)
        return None

    async def execute(self, query: str, *args):
        if "UPDATE pr_reviews" in query and "SET pr_context" in query:
            context_json, review_id = args
            row = self.reviews.get(review_id)
            if row:
                row["pr_context"] = json.loads(context_json) if isinstance(context_json, str) else context_json
                row["lock_version"] += 1
                row["updated_at"] = datetime.now(timezone.utc)
                return "UPDATE 1"
        elif "UPDATE review_jobs" in query and "SET review_id" in query:
            job_id, review_id = args
            job = self.jobs.get(job_id)
            if job:
                job["review_id"] = review_id
                return "UPDATE 1"
        elif "UPDATE pr_reviews" in query and "SET status" in query:
            status, result_json, review_id = args[0], args[1], args[-1]
            row = self.reviews.get(review_id)
            if row:
                row["status"] = status
                row["result"] = json.loads(result_json) if isinstance(result_json, str) else result_json
                row["lock_version"] += 1
                row["updated_at"] = datetime.now(timezone.utc)
                return "UPDATE 1"
        elif "UPDATE pr_reviews" in query and "SET result" in query:
            result_json, review_id = args[0], args[-1]
            row = self.reviews.get(review_id)
            if row:
                row["result"] = json.loads(result_json) if isinstance(result_json, str) else result_json
                row["lock_version"] += 1
                row["updated_at"] = datetime.now(timezone.utc)
                return "UPDATE 1"
        return "UPDATE 0"

    async def fetch(self, query: str, *args):
        rows = list(self.reviews.values())
        rows.sort(key=lambda item: item["id"], reverse=True)
        return [FakeRecord(row) for row in rows[:50]]

    async def close(self):
        return None

    def seed_review(
        self,
        review_id: int,
        repo: str,
        pr_number: int,
        status: str,
        pr_context: dict,
        result: dict | None = None,
    ) -> int:
        now = datetime.now(timezone.utc)
        self.reviews[review_id] = {
            "id": review_id,
            "repo": repo,
            "pr_number": pr_number,
            "status": status,
            "pr_context": pr_context,
            "result": result,
            "source_sha": pr_context.get("source_sha", f"seed-{review_id}"),
            "review_version": pr_context.get("review_version", 1),
            "webhook_delivery_id": None,
            "lock_version": 0,
            "created_at": now,
            "updated_at": now,
        }
        self._next_id = max(self._next_id, review_id + 1)
        return review_id


class FakeContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeGraph:
    def invoke(self, _input=None, config=None):
        return {
            "pr_summary": "Mock summary",
            "ranked_findings": [],
            "merge_decision": {
                "decision": "approve",
                "reason": "No significant issues found.",
            },
            "grounding_summary": {"verified": 0, "dropped": 0},
        }

    def get_state_history(self, config):
        return []
