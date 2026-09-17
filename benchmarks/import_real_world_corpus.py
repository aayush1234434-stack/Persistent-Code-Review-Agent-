"""Import a pinned, unique-PR sample from the public GitHub review dataset."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "ronantakizawa/github-codereview"
DATASET_REVISION = "c3e3c6e7e9f61e3e7a5b52894bcd440d586ae6ca"
DATASET_API = "https://datasets-server.huggingface.co/rows"
DATASET_METADATA_API = f"https://huggingface.co/api/datasets/{DATASET}"
DEFAULT_CATEGORY_QUOTAS = {
    "logic": 60,
    "tests": 15,
    "performance": 10,
    "security": 10,
    "concurrency": 10,
    "contract": 10,
    "data": 5,
}


def infer_category(comment: str) -> str:
    text = comment.lower()
    category_terms = (
        ("security", ("security", "auth", "permission", "token", "password", "secret", "xss", "csrf", "injection")),
        ("concurrency", ("race", "thread", "lock", "concurr", "deadlock", "atomic")),
        ("performance", ("performance", "slow", "cache", "query", "memory", "latency", "computation", "optimiz")),
        ("tests", ("test", "assert", "coverage", "fixture", "mock")),
        ("contract", ("api", "schema", "backward", "compatib", "interface", "signature", "public method")),
        ("data", ("data loss", "corrupt", "serializ", "deserializ", "parse", "validat", "database")),
    )
    for category, terms in category_terms:
        if any(term in text for term in terms):
            return category
    return "logic"


def fetch_json(url: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "persistent-code-review-benchmark/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                retry_after = 0.0
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                    try:
                        retry_after = float(exc.headers.get("Retry-After", 0))
                    except (TypeError, ValueError):
                        retry_after = 0.0
                time.sleep(min(max(retry_after, 1.0 * attempt), 20.0))
    raise RuntimeError(f"Unable to fetch {url}: {last_error}") from last_error


def verify_dataset_revision() -> None:
    current_revision = str(fetch_json(DATASET_METADATA_API).get("sha", ""))
    if current_revision != DATASET_REVISION:
        raise RuntimeError(
            "Upstream dataset revision changed; review the source and update "
            f"DATASET_REVISION intentionally (expected {DATASET_REVISION}, got {current_revision})"
        )


def fetch_rows(offset: int, length: int = 100) -> list[dict[str, Any]]:
    url = DATASET_API + "?" + urllib.parse.urlencode({
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length,
    })
    return fetch_json(url)["rows"]


def normalize_case(item: dict[str, Any]) -> dict[str, Any]:
    row = item["row"]
    repository = str(row["repo_name"])
    pr_number = int(row["pr_number"])
    category = infer_category(str(row.get("reviewer_comment", "")))
    case_id = f"github-{repository.replace('/', '-')}-{pr_number}"
    return {
        "id": case_id,
        "source": {
            "type": "real_github_pr_human_review",
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "dataset_row": int(item["row_idx"]),
            "repository": repository,
            "pr_number": pr_number,
            "url": f"https://github.com/{repository}/pull/{pr_number}",
        },
        "metadata": {
            "title": str(row.get("pr_title", "")),
            "language": str(row.get("language") or row.get("repo_language") or "unknown"),
            "file": str(row.get("file_path", "")),
            "quality_score": float(row.get("quality_score", 0.0)),
            "source_comment_line": int(row["comment_line"]) if row.get("comment_line") is not None else None,
        },
        "diff": str(row.get("diff_context", ""))[:6000],
        "labels": [{
            "id": f"{case_id}-human-review-1",
            "category": category,
            "file": str(row.get("file_path", "")),
            "line": None,
            "description": str(row.get("reviewer_comment", ""))[:2000],
            "human_verified": True,
        }],
    }


def import_cases(limit: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_prs: set[tuple[str, int]] = set()
    quotas = dict(DEFAULT_CATEGORY_QUOTAS) if limit == sum(DEFAULT_CATEGORY_QUOTAS.values()) else {}
    overflow: list[dict[str, Any]] = []
    for offset in range(0, 20_000, 100):
        rows = fetch_rows(offset)
        if not rows:
            break
        for item in rows:
            row = item["row"]
            key = (str(row.get("repo_name")), int(row.get("pr_number", 0)))
            if row.get("is_negative") or key in seen_prs:
                continue
            seen_prs.add(key)
            normalized = normalize_case(item)
            category = normalized["labels"][0]["category"]
            if quotas:
                if sum(1 for case in cases if case["labels"][0]["category"] == category) < quotas.get(category, 0):
                    cases.append(normalized)
                else:
                    overflow.append(normalized)
            else:
                cases.append(normalized)
            if len(cases) >= limit:
                return sorted(cases, key=lambda case: case["id"])
    if len(cases) < limit and overflow:
        cases.extend(overflow[: limit - len(cases)])
    if len(cases) >= limit:
        return sorted(cases[:limit], key=lambda case: case["id"])
    raise RuntimeError(f"Dataset ended after {len(cases)} unique positive PRs; requested {limit}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "corpus" / "real_world_prs.v1.jsonl",
    )
    args = parser.parse_args()
    verify_dataset_revision()
    cases = import_cases(args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases))
    print(json.dumps({"output": str(args.output), "cases": len(cases), "dataset_revision": DATASET_REVISION}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
