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
        self._next_id = 1

    async def fetchval(self, query: str, *args):
        if "INSERT INTO pr_reviews" in query:
            review_id = self._next_id
            self._next_id += 1
            pr_context = args[3]
            if isinstance(pr_context, str):
                pr_context = json.loads(pr_context)
            now = datetime.now(timezone.utc)
            self.reviews[review_id] = {
                "id": review_id,
                "repo": args[0],
                "pr_number": args[1],
                "status": args[2],
                "pr_context": pr_context,
                "result": None,
                "created_at": now,
                "updated_at": now,
            }
            return review_id
        if "SELECT 1" in query:
            return 1
        return None

    async def fetchrow(self, query: str, *args):
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
                    and ctx.get("source_sha") == source_sha
                ):
                    return FakeRecord({"id": row["id"], "status": row["status"]})
        return None

    async def execute(self, query: str, *args):
        if "UPDATE pr_reviews" in query and "SET status" in query:
            status, result_json, review_id = args[0], args[1], args[-1]
            row = self.reviews.get(review_id)
            if row:
                row["status"] = status
                row["result"] = json.loads(result_json) if isinstance(result_json, str) else result_json
                row["updated_at"] = datetime.now(timezone.utc)
        elif "UPDATE pr_reviews" in query and "SET result" in query:
            result_json, review_id = args[0], args[-1]
            row = self.reviews.get(review_id)
            if row:
                row["result"] = json.loads(result_json) if isinstance(result_json, str) else result_json
                row["updated_at"] = datetime.now(timezone.utc)

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
            "created_at": now,
            "updated_at": now,
        }
        self._next_id = max(self._next_id, review_id + 1)
        return review_id


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
