import hmac
import hashlib
import json
import os
import asyncio
import importlib
import logging
import traceback
import time
from enum import Enum
from typing import Any
import httpx
import asyncpg
from agent import build_graph
from fastapi import FastAPI, Request, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
DASHBOARD_API_KEY = os.environ.get("DASHBOARD_API_KEY")
HTTP_TIMEOUT_SECONDS = float(os.environ.get("HTTP_TIMEOUT_SECONDS", "20"))


class ReviewStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"

# -----------------------------
# PR size protection
# -----------------------------
MAX_FILES = 20
MAX_LINES_PER_FILE = 200
MAX_TOTAL_LINES = 1500
REVIEW_RULES_PATH = ".github/pr-reviewer.yml"
DEFAULT_PROMPT_CHAR_BUDGET = int(os.environ.get("REVIEW_PROMPT_CHAR_BUDGET", "12000"))


# -----------------------------
# Verify GitHub webhook signature
# -----------------------------
def verify_github_signature(raw_body: bytes, signature: str) -> bool:
    if not GITHUB_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Webhook secret is not configured")
    expected_signature = "sha256=" + hmac.new(
        GITHUB_WEBHOOK_SECRET.encode(),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected_signature, signature)


# -----------------------------
# Fetch PR diff
# -----------------------------
async def get_pr_diff(repo: str, pr_number: int) -> str:
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=503, detail="GitHub token is not configured")
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.diff",
    }

    async def _request():
        timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.get(url, headers=headers)

    try:
        response = await github_request_with_retry(_request)
        response.raise_for_status()
        return response.text
    except httpx.HTTPStatusError as e:
        logger.exception("Failed fetching PR diff", extra={"repo": repo, "pr_number": pr_number})
        raise HTTPException(
            status_code=502,
            detail=f"GitHub API error: {e.response.status_code}",
        )


async def get_repo_review_rules(repo: str, ref: str | None = None) -> dict[str, Any]:
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=503, detail="GitHub token is not configured")
    url = f"https://api.github.com/repos/{repo}/contents/{REVIEW_RULES_PATH}"
    params = {"ref": ref} if ref else None
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.raw",
    }

    async def _request():
        timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.get(url, headers=headers, params=params)

    response = await github_request_with_retry(_request)
    if response.status_code == 404:
        return {}
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.warning(
            "Failed fetching repo review rules",
            extra={"repo": repo, "status_code": e.response.status_code},
        )
        return {}
    return parse_review_rules(response.text)


def parse_review_rules(raw_rules: str) -> dict[str, Any]:
    if not raw_rules.strip():
        return {}
    try:
        yaml = importlib.import_module("yaml")
        parsed = yaml.safe_load(raw_rules)
    except ModuleNotFoundError:
        try:
            parsed = json.loads(raw_rules)
        except json.JSONDecodeError:
            logger.warning("PyYAML unavailable and review rules are not JSON; ignoring rules")
            return {}
    except Exception as exc:
        logger.warning("Unable to parse review rules: %s", exc)
        return {}
    return parsed if isinstance(parsed, dict) else {}


# -----------------------------
# Detect file type
# -----------------------------
def detect_file_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()

    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".go": "go",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
    }

    return mapping.get(ext, "unknown")


# -----------------------------
# Parse git diff
# -----------------------------
def parse_diff(raw_diff: str) -> list[dict]:
    files = []
    current_file = None
    current_chunk = None
    old_line_no = None
    new_line_no = None
    diff_position = 0

    for line in raw_diff.splitlines():

        # New file begins
        if line.startswith("diff --git"):

            if current_chunk and current_file:
                current_file["chunks"].append(current_chunk)

            if current_file and current_file["filename"]:
                files.append(current_file)

            current_file = {
                "filename": "",
                "file_type": "",
                "change_type": "modified",
                "added_lines": [],
                "removed_lines": [],
                "added_line_details": [],
                "removed_line_details": [],
                "chunks": [],
            }

            current_chunk = None
            old_line_no = None
            new_line_no = None
            diff_position = 0

        # File added
        elif line.startswith("new file mode") and current_file:
            current_file["change_type"] = "added"

        # File deleted
        elif line.startswith("deleted file mode") and current_file:
            current_file["change_type"] = "deleted"

        # File path — handle deleted files ("+++ /dev/null") via "--- a/" instead
        elif line.startswith("--- a/") and current_file and current_file["change_type"] == "deleted":
            filename = line[6:]
            current_file["filename"] = filename
            current_file["file_type"] = detect_file_type(filename)

        elif line.startswith("+++ b/") and current_file:
            filename = line[6:]
            current_file["filename"] = filename
            current_file["file_type"] = detect_file_type(filename)

        # New chunk
        elif line.startswith("@@") and current_file:

            if current_chunk:
                current_file["chunks"].append(current_chunk)

            old_line_no = None
            new_line_no = None
            try:
                header = line.split("@@")[1].strip()
                header_parts = header.split(" ")
                old_part = next((part for part in header_parts if part.startswith("-")), None)
                new_part = next((part for part in header_parts if part.startswith("+")), None)
                if old_part:
                    old_line_no = int(old_part[1:].split(",")[0])
                if new_part:
                    new_line_no = int(new_part[1:].split(",")[0])
            except Exception:
                old_line_no = None
                new_line_no = None

            current_chunk = {
                "added": [],
                "removed": [],
                "added_line_details": [],
                "removed_line_details": [],
            }

        # Added lines
        elif current_file and line.startswith("+") and not line.startswith("+++"):
            clean = line[1:]
            current_file["added_lines"].append(clean)
            current_file["added_line_details"].append({
                "line": new_line_no,
                "content": clean,
                "diff_position": diff_position + 1,
            })

            if current_chunk is None:
                current_chunk = {
                    "added": [],
                    "removed": [],
                    "added_line_details": [],
                    "removed_line_details": [],
                }

            current_chunk["added"].append(clean)
            current_chunk["added_line_details"].append({
                "line": new_line_no,
                "content": clean,
                "diff_position": diff_position + 1,
            })
            if new_line_no is not None:
                new_line_no += 1
            diff_position += 1

        # Removed lines
        elif current_file and line.startswith("-") and not line.startswith("---"):
            clean = line[1:]
            current_file["removed_lines"].append(clean)
            current_file["removed_line_details"].append({
                "line": old_line_no,
                "content": clean,
                "diff_position": diff_position + 1,
            })

            if current_chunk is None:
                current_chunk = {
                    "added": [],
                    "removed": [],
                    "added_line_details": [],
                    "removed_line_details": [],
                }

            current_chunk["removed"].append(clean)
            current_chunk["removed_line_details"].append({
                "line": old_line_no,
                "content": clean,
                "diff_position": diff_position + 1,
            })
            if old_line_no is not None:
                old_line_no += 1
            diff_position += 1
        elif current_file:
            if old_line_no is not None:
                old_line_no += 1
            if new_line_no is not None:
                new_line_no += 1
            if current_chunk is not None:
                diff_position += 1

    # Final flush
    if current_chunk and current_file:
        current_file["chunks"].append(current_chunk)

    if current_file and current_file["filename"]:
        files.append(current_file)

    return files


# -----------------------------
# Prune huge PRs
# -----------------------------
def prune_diff(parsed_diff: list[dict]) -> list[dict]:
    pruned, _budget = prune_diff_with_budget(parsed_diff)
    return pruned


def review_cost_controls(review_rules: dict | None) -> dict:
    rules = review_rules or {}
    controls = rules.get("cost_controls") or rules.get("budgets") or {}
    if not isinstance(controls, dict):
        controls = {}

    def positive_int(name: str, default: int) -> int:
        try:
            value = int(controls.get(name, default))
        except Exception:
            return default
        return value if value > 0 else default

    return {
        "max_files": positive_int("max_files", MAX_FILES),
        "max_lines_per_file": positive_int("max_lines_per_file", MAX_LINES_PER_FILE),
        "max_total_lines": positive_int("max_total_lines", MAX_TOTAL_LINES),
        "prompt_char_budget": positive_int("prompt_char_budget", DEFAULT_PROMPT_CHAR_BUDGET),
        "triage_model": str(controls.get("triage_model") or os.environ.get("OPENAI_TRIAGE_MODEL", "gpt-4o-mini")),
        "strong_model": str(controls.get("strong_model") or os.environ.get("OPENAI_STRONG_REVIEW_MODEL", os.environ.get("OPENAI_REVIEW_MODEL", "gpt-4o"))),
        "strong_model_file_risk": str(controls.get("strong_model_file_risk") or "high"),
        "strong_model_min_changed_lines": positive_int("strong_model_min_changed_lines", 400),
    }


def staged_review_model_policy(parsed_diff: list[dict], review_rules: dict | None, cost_controls: dict) -> dict:
    risky_extensions = set((review_rules or {}).get("risky_extensions", [".py", ".js", ".ts", ".sql", ".go"]))
    risky_files = [
        file.get("filename", "")
        for file in parsed_diff
        if os.path.splitext(file.get("filename", ""))[1].lower() in risky_extensions
    ]
    changed_lines = sum(len(file.get("added_lines", [])) + len(file.get("removed_lines", [])) for file in parsed_diff)
    uses_strong = bool(risky_files) and changed_lines >= int(cost_controls["strong_model_min_changed_lines"])
    return {
        "triage_model": cost_controls["triage_model"],
        "strong_model": cost_controls["strong_model"],
        "selected_model": cost_controls["strong_model"] if uses_strong else cost_controls["triage_model"],
        "uses_strong_model": uses_strong,
        "risky_files": risky_files[:25],
        "changed_lines": changed_lines,
        "reason": "risky files and size threshold met" if uses_strong else "cheap triage model is sufficient for configured budget",
    }


def prune_diff_with_budget(
    parsed_diff: list[dict],
    max_files: int = MAX_FILES,
    max_lines_per_file: int = MAX_LINES_PER_FILE,
    max_total_lines: int = MAX_TOTAL_LINES,
) -> tuple[list[dict], dict]:
    pruned_files = []
    total_lines = 0
    pruned_files_info = []

    for file in parsed_diff[:max_files]:

        added = file["added_lines"][:max_lines_per_file]
        removed = file["removed_lines"][:max_lines_per_file]

        chunks = []
        running_lines = 0

        for chunk in file["chunks"]:

            chunk_added = chunk["added"][:max_lines_per_file]
            chunk_removed = chunk["removed"][:max_lines_per_file]
            chunk_added_details = chunk.get("added_line_details", [])[:max_lines_per_file]
            chunk_removed_details = chunk.get("removed_line_details", [])[:max_lines_per_file]

            chunk_size = len(chunk_added) + len(chunk_removed)

            if total_lines + running_lines + chunk_size > max_total_lines:
                pruned_files_info.append({
                    "filename": file.get("filename", ""),
                    "reason": "total line budget reached",
                })
                break

            chunks.append({
                "added": chunk_added,
                "removed": chunk_removed,
                "added_line_details": chunk_added_details,
                "removed_line_details": chunk_removed_details,
            })

            running_lines += chunk_size

        original_line_count = len(file.get("added_lines", [])) + len(file.get("removed_lines", []))
        kept_line_count = len(added) + len(removed)
        if kept_line_count < original_line_count:
            pruned_files_info.append({
                "filename": file.get("filename", ""),
                "reason": "per-file line budget reached",
                "kept_lines": kept_line_count,
                "original_lines": original_line_count,
            })

        file_copy = {
            **file,
            "added_lines": added,
            "removed_lines": removed,
            "added_line_details": file.get("added_line_details", [])[:max_lines_per_file],
            "removed_line_details": file.get("removed_line_details", [])[:max_lines_per_file],
            "chunks": chunks,
        }

        # Fix: count only lines from kept chunks, not all added/removed lines
        total_lines += running_lines

        if total_lines >= max_total_lines:
            pruned_files.append(file_copy)
            break

        pruned_files.append(file_copy)

    skipped_files = [
        {
            "filename": file.get("filename", ""),
            "reason": "file count budget reached",
        }
        for file in parsed_diff[max_files:]
    ]

    budget = {
        "max_files": max_files,
        "max_lines_per_file": max_lines_per_file,
        "max_total_lines": max_total_lines,
        "total_files_seen": len(parsed_diff),
        "total_files_included": len(pruned_files),
        "total_files_skipped": len(skipped_files),
        "total_changed_lines_included": total_lines,
        "skipped_files": skipped_files,
        "pruned_files": pruned_files_info,
        "truncated": bool(skipped_files or pruned_files_info),
    }

    return pruned_files, budget


# -----------------------------
# Build PR context
# -----------------------------
def build_pr_context(metadata: dict, raw_diff: str, review_rules: dict | None = None) -> dict:
    full_parsed_diff = parse_diff(raw_diff)
    cost_controls = review_cost_controls(review_rules)
    model_policy = staged_review_model_policy(full_parsed_diff, review_rules, cost_controls)
    parsed_diff, review_budget = prune_diff_with_budget(
        full_parsed_diff,
        max_files=cost_controls["max_files"],
        max_lines_per_file=cost_controls["max_lines_per_file"],
        max_total_lines=cost_controls["max_total_lines"],
    )
    prompt_char_budget = cost_controls["prompt_char_budget"]
    prompt_chars_estimate = len(json.dumps(parsed_diff, ensure_ascii=False))
    review_budget = {
        **review_budget,
        "prompt_char_budget": prompt_char_budget,
        "prompt_chars_estimate": prompt_chars_estimate,
        "prompt_truncated": prompt_chars_estimate > prompt_char_budget,
        "cost_controls": cost_controls,
    }

    return {
        "pr_number": metadata["pr_number"],
        "title": metadata["title"],
        "description": metadata["description"] or "No description provided",
        "author": metadata["author"],
        "action": metadata["action"],
        "url": metadata["url"],
        "source_branch": metadata["source_branch"],
        "source_sha": metadata.get("source_sha"),
        "target_branch": metadata["target_branch"],
        "target_sha": metadata.get("target_sha"),
        "repository": metadata["repository"],
        "total_files_changed": len(parsed_diff),
        "files": parsed_diff,
        "review_rules": review_rules or {},
        "review_rules_path": REVIEW_RULES_PATH,
        "review_budget": review_budget,
        "review_model_policy": model_policy,
        "human_feedback_memory": metadata.get("human_feedback_memory", {}),
    }


def build_queued_pr_context(metadata: dict) -> dict:
    return {
        "pr_number": metadata["pr_number"],
        "title": metadata["title"],
        "description": metadata["description"] or "No description provided",
        "author": metadata["author"],
        "action": metadata["action"],
        "url": metadata["url"],
        "source_branch": metadata["source_branch"],
        "source_sha": metadata.get("source_sha"),
        "target_branch": metadata["target_branch"],
        "target_sha": metadata.get("target_sha"),
        "repository": metadata["repository"],
        "draft": bool(metadata.get("draft")),
        "total_files_changed": 0,
        "files": [],
        "review_rules": {},
        "review_rules_path": REVIEW_RULES_PATH,
        "review_budget": {
            "max_files": MAX_FILES,
            "max_lines_per_file": MAX_LINES_PER_FILE,
            "max_total_lines": MAX_TOTAL_LINES,
            "total_files_seen": 0,
            "total_files_included": 0,
            "total_files_skipped": 0,
            "total_changed_lines_included": 0,
            "skipped_files": [],
            "pruned_files": [],
            "truncated": False,
            "prompt_char_budget": DEFAULT_PROMPT_CHAR_BUDGET,
            "prompt_chars_estimate": 0,
            "prompt_truncated": False,
        },
        "review_model_policy": {
            "triage_model": os.environ.get("OPENAI_TRIAGE_MODEL", "gpt-4o-mini"),
            "strong_model": os.environ.get("OPENAI_STRONG_REVIEW_MODEL", os.environ.get("OPENAI_REVIEW_MODEL", "gpt-4o")),
            "selected_model": os.environ.get("OPENAI_TRIAGE_MODEL", "gpt-4o-mini"),
            "uses_strong_model": False,
            "risky_files": [],
            "changed_lines": 0,
            "reason": "draft PR queued before analysis",
        },
    }


def extract_pr_metadata(payload: dict) -> dict:
    pr = payload["pull_request"]
    return {
        "action": payload["action"],
        "repository": payload["repository"]["full_name"],
        "pr_number": pr["number"],
        "title": pr["title"],
        "description": pr["body"],
        "author": pr["user"]["login"],
        "state": pr["state"],
        "url": pr["html_url"],
        "source_branch": pr["head"]["ref"],
        "source_sha": pr["head"]["sha"],
        "target_branch": pr["base"]["ref"],
        "target_sha": pr["base"]["sha"],
        "created_at": pr["created_at"],
        "updated_at": pr["updated_at"],
        "merged": pr.get("merged"),
        "draft": bool(pr.get("draft")),
    }


# -----------------------------
# Post review comment to GitHub PR
# -----------------------------
async def post_pr_comment(repo: str, pr_number: int, body: str) -> None:
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=503, detail="GitHub token is not configured")
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    async def _request():
        timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(url, headers=headers, json={"body": body})

    try:
        response = await github_request_with_retry(_request)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.exception("Failed posting PR comment", extra={"repo": repo, "pr_number": pr_number})
        raise HTTPException(
            status_code=502,
            detail=f"Failed to post PR comment: {e.response.status_code}",
        )


async def post_inline_review_comment(
    repo: str,
    pr_number: int,
    commit_id: str,
    path: str,
    line: int,
    body: str,
    position: int | None = None,
) -> None:
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=503, detail="GitHub token is not configured")
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {
        "body": body,
        "commit_id": commit_id,
        "path": path,
    }
    if position is not None:
        payload["position"] = position
    else:
        payload["line"] = line
        payload["side"] = "RIGHT"

    async def _request():
        timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(url, headers=headers, json=payload)

    response = await github_request_with_retry(_request)
    response.raise_for_status()


def diff_position_for_finding(pr_context: dict, finding: dict) -> int | None:
    file_name = finding.get("file")
    line = finding.get("line")
    if not file_name or not isinstance(line, int):
        return None
    for file in pr_context.get("files", []):
        if file.get("filename") != file_name:
            continue
        for detail in file.get("added_line_details", []):
            if detail.get("line") == line and detail.get("diff_position") is not None:
                return int(detail["diff_position"])
        for detail in file.get("removed_line_details", []):
            if detail.get("line") == line and detail.get("diff_position") is not None:
                return int(detail["diff_position"])
    return None


async def post_inline_review_comments(repo: str, pr_number: int, pr_context: dict, findings: list[dict]) -> dict:
    commit_id = pr_context.get("source_sha")
    if not commit_id:
        return {"posted": 0, "skipped": len(findings), "reason": "missing source_sha"}

    posted = 0
    skipped = 0
    errors = []
    for finding in findings[:10]:
        path = finding.get("file")
        line = finding.get("line")
        position = diff_position_for_finding(pr_context, finding)
        if not path or not isinstance(line, int):
            skipped += 1
            continue
        if position is None:
            skipped += 1
            continue
        body = (
            f"Automated review finding ({finding.get('severity', 'unknown')}, "
            f"{finding.get('finding_type', 'possible_concern')}, "
            f"confidence {finding.get('confidence', 'n/a')}):\n\n"
            f"{finding.get('description', 'No description')}\n\n"
            f"Evidence: {finding.get('evidence', 'changed diff line')}"
        )
        try:
            await post_inline_review_comment(repo, pr_number, commit_id, path, line, body, position=position)
            posted += 1
        except Exception as exc:
            skipped += 1
            errors.append({"file": path, "line": line, "error": str(exc)})
            logger.warning("Failed posting inline review comment: %s", exc)
    return {"posted": posted, "skipped": skipped, "errors": errors}


async def github_request_with_retry(request_func, retries: int = 3) -> httpx.Response:
    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            response = await request_func()
            if response.status_code == 403 and "x-ratelimit-reset" in response.headers and attempt < retries:
                try:
                    reset_epoch = int(response.headers["x-ratelimit-reset"])
                    now_epoch = int(time.time())
                    await asyncio.sleep(min(max(reset_epoch - now_epoch, 1), 10))
                    continue
                except Exception:
                    await asyncio.sleep(0.5 * attempt)
                    continue
            if response.status_code >= 500 and attempt < retries:
                await asyncio.sleep(0.4 * attempt)
                continue
            return response
        except httpx.TransportError as exc:
            last_exception = exc
            if attempt < retries:
                await asyncio.sleep(0.4 * attempt)
                continue
            raise HTTPException(status_code=502, detail=f"GitHub transport error: {exc}") from exc
    if last_exception is not None:
        raise HTTPException(status_code=502, detail="GitHub request failed")
    raise HTTPException(status_code=502, detail="GitHub request failed")


async def save_review(pool: asyncpg.Pool, pr_context: dict, status: str) -> int:
    return await pool.fetchval(
        """
        INSERT INTO pr_reviews (repo, pr_number, status, pr_context)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        """,
        pr_context["repository"],
        pr_context["pr_number"],
        status,
        json.dumps(pr_context),
    )


async def get_existing_review_for_event(pool: asyncpg.Pool, pr_context: dict) -> asyncpg.Record | None:
    return await pool.fetchrow(
        """
        SELECT id, status
        FROM pr_reviews
        WHERE repo = $1
          AND pr_number = $2
          AND pr_context->>'source_sha' = $3
        ORDER BY id DESC
        LIMIT 1
        """,
        pr_context["repository"],
        pr_context["pr_number"],
        pr_context.get("source_sha"),
    )


async def update_review(
    pool: asyncpg.Pool, review_id: int, status: str, result: dict
) -> None:
    await pool.execute(
        """
        UPDATE pr_reviews
        SET status = $1, result = $2, updated_at = NOW()
        WHERE id = $3
        """,
        status,
        json.dumps(result),
        review_id,
    )


async def update_review_result(pool: asyncpg.Pool, review_id: int, result: dict) -> None:
    await pool.execute(
        """
        UPDATE pr_reviews
        SET result = $1, updated_at = NOW()
        WHERE id = $2
        """,
        json.dumps(result),
        review_id,
    )


async def get_review(pool: asyncpg.Pool, review_id: int) -> asyncpg.Record | None:
    return await pool.fetchrow(
        """
        SELECT id, repo, pr_number, status, pr_context, result, created_at, updated_at
        FROM pr_reviews
        WHERE id = $1
        """,
        review_id,
    )


async def list_review_rows(
    pool: asyncpg.Pool,
    repo: str | None = None,
    pr_number: int | None = None,
    status: str | None = None,
    author: str | None = None,
    decision: str | None = None,
    search: str | None = None,
) -> list[asyncpg.Record]:
    conditions = []
    values: list[Any] = []

    def add_value(value: Any) -> str:
        values.append(value)
        return f"${len(values)}"

    if repo:
        conditions.append(f"repo ILIKE {add_value('%' + repo + '%')}")
    if pr_number is not None:
        conditions.append(f"pr_number = {add_value(pr_number)}")
    if status:
        conditions.append(f"status = {add_value(status)}")
    if author:
        conditions.append(f"pr_context->>'author' ILIKE {add_value('%' + author + '%')}")
    if decision:
        conditions.append(f"result->'merge_decision'->>'decision' = {add_value(decision)}")
    if search:
        token = "%" + search + "%"
        placeholder = add_value(token)
        conditions.append(
            f"(repo ILIKE {placeholder} OR pr_context->>'title' ILIKE {placeholder} OR pr_context->>'author' ILIKE {placeholder})"
        )

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return await pool.fetch(
        f"""
        SELECT id, repo, pr_number, status, result, pr_context, created_at, updated_at
        FROM pr_reviews
        {where}
        ORDER BY id DESC
        LIMIT 100
        """,
        *values,
    )


def format_review_comment(result: dict) -> str:
    decision = result.get("merge_decision", {}).get("decision", "needs_review")
    reason = result.get("merge_decision", {}).get("reason", "No reason provided.")
    findings = result.get("ranked_findings", [])[:5]
    budget = result.get("review_budget", {})
    grounding = result.get("grounding_summary", {})

    lines = [
        "## 🤖 Automated PR Review",
        f"**Decision:** `{decision}`",
        f"**Reason:** {reason}",
        "",
        f"**Grounding:** {grounding.get('verified', 0)} verified, {grounding.get('dropped', 0)} dropped as ungrounded.",
    ]

    if budget:
        lines.extend([
            f"**Diff budget:** {budget.get('total_files_included', 0)}/{budget.get('total_files_seen', 0)} files included; "
            f"diff truncated: `{budget.get('truncated', False)}`; "
            f"prompt truncated: `{budget.get('prompt_truncated', False)}`.",
        ])
        controls = budget.get("cost_controls", {})
        if controls:
            lines.append(
                f"**Cost controls:** max files `{controls.get('max_files')}`, "
                f"max total lines `{controls.get('max_total_lines')}`, "
                f"prompt chars `{controls.get('prompt_char_budget')}`."
            )
        if budget.get("skipped_files") or budget.get("pruned_files"):
            lines.append("Some files or lines were skipped/pruned before model review.")

    lines.extend([
        "",
        "### Top Findings",
    ])

    if not findings:
        lines.append("- No significant issues found.")
    else:
        for finding in findings:
            sev = finding.get("severity", "unknown")
            cat = finding.get("category", "general")
            desc = finding.get("description", "No description")
            file_name = finding.get("file", "unknown file")
            confidence = finding.get("confidence", "n/a")
            finding_type = finding.get("finding_type", "possible_concern")
            line = finding.get("line")
            location = f"{file_name}:{line}" if line else file_name
            lines.append(f"- [{sev}/{cat}/{finding_type}/confidence={confidence}] {desc} (`{location}`)")
            if finding.get("evidence"):
                lines.append(f"  - Evidence: {finding.get('evidence')}")
            context = finding.get("line_context", [])[:3]
            for item in context:
                lines.append(
                    f"  - `{item.get('change', 'changed')}:{item.get('line')}` {item.get('content', '')[:160]}"
                )

    return "\n".join(lines)


def finding_identity(finding: dict) -> str:
    return "|".join([
        str(finding.get("file", "")),
        str(finding.get("line", "")),
        str(finding.get("description", "")).strip().lower()[:120],
    ])


def compare_findings(old_findings: list[dict], new_findings: list[dict]) -> dict:
    old_map = {finding_identity(finding): finding for finding in old_findings if isinstance(finding, dict)}
    new_map = {finding_identity(finding): finding for finding in new_findings if isinstance(finding, dict)}
    old_keys = set(old_map)
    new_keys = set(new_map)
    return {
        "added": [new_map[key] for key in sorted(new_keys - old_keys)],
        "removed": [old_map[key] for key in sorted(old_keys - new_keys)],
        "unchanged": [new_map[key] for key in sorted(old_keys & new_keys)],
        "counts": {
            "old": len(old_map),
            "new": len(new_map),
            "added": len(new_keys - old_keys),
            "removed": len(old_keys - new_keys),
            "unchanged": len(old_keys & new_keys),
        },
    }


def review_timeline(row: asyncpg.Record, result: dict) -> list[dict]:
    pr_context = as_dict(row["pr_context"])
    timeline = [
        {
            "label": "Webhook received",
            "status": "completed",
            "at": row["created_at"].isoformat() if row["created_at"] else None,
            "detail": f"{pr_context.get('action', 'unknown')} event for {row['repo']}#{row['pr_number']}",
        }
    ]
    if pr_context.get("draft"):
        timeline.append({
            "label": "Draft PR queued",
            "status": "completed" if row["status"] == ReviewStatus.QUEUED.value else "completed",
            "at": row["created_at"].isoformat() if row["created_at"] else None,
            "detail": "Analysis is held until GitHub sends ready_for_review.",
        })
    if row["status"] != ReviewStatus.QUEUED.value and (result.get("pr_summary") or "ranked_findings" in result):
        timeline.append({
            "label": "Analysis complete",
            "status": "completed" if result.get("ranked_findings") is not None else "pending",
            "at": row["updated_at"].isoformat() if row["updated_at"] else None,
            "detail": f"{len(result.get('ranked_findings', []))} grounded finding(s).",
        })
    if row["status"] == ReviewStatus.AWAITING_APPROVAL.value:
        timeline.append({
            "label": "Awaiting reviewer action",
            "status": "current",
            "at": row["updated_at"].isoformat() if row["updated_at"] else None,
            "detail": "Reviewer can post final review, rerun analysis, or mark findings.",
        })
    if result.get("human_decision"):
        timeline.append({
            "label": "Human decision",
            "status": "completed",
            "at": row["updated_at"].isoformat() if row["updated_at"] else None,
            "detail": result["human_decision"].get("decision", "recorded"),
        })
    if result.get("final_review_posted"):
        timeline.append({
            "label": "Final review posted",
            "status": "completed",
            "at": result["final_review_posted"].get("posted_at"),
            "detail": result["final_review_posted"].get("summary", ""),
        })
    if row["status"] == ReviewStatus.FAILED.value:
        timeline.append({
            "label": "Failed",
            "status": "failed",
            "at": row["updated_at"].isoformat() if row["updated_at"] else None,
            "detail": result.get("error", "Review failed."),
        })
    return timeline


def feedback_summary(result: dict) -> dict:
    feedback = result.get("finding_feedback", {})
    valid = sum(1 for item in feedback.values() if item.get("verdict") == "valid")
    invalid = sum(1 for item in feedback.values() if item.get("verdict") == "invalid")
    return {"valid": valid, "invalid": invalid, "total": len(feedback)}


async def summarize_human_feedback(pool: asyncpg.Pool, repo: str, author: str | None = None) -> dict:
    rows = await pool.fetch(
        """
        SELECT result
        FROM pr_reviews
        WHERE repo = $1
          AND result ? 'finding_feedback'
        ORDER BY id DESC
        LIMIT 100
        """,
        repo,
    )
    category_counts = {"valid": {}, "invalid": {}}
    file_counts = {"valid": {}, "invalid": {}}
    for row in rows:
        result = as_dict(row["result"])
        findings = result.get("ranked_findings", [])
        for key, feedback in result.get("finding_feedback", {}).items():
            try:
                finding = findings[int(key)]
            except Exception:
                finding = {}
            verdict = feedback.get("verdict")
            if verdict not in category_counts:
                continue
            category = finding.get("category", "general")
            file_name = finding.get("file", "unknown")
            category_counts[verdict][category] = category_counts[verdict].get(category, 0) + 1
            file_counts[verdict][file_name] = file_counts[verdict].get(file_name, 0) + 1

    def frequent_keys(counts: dict[str, int]) -> list[str]:
        return [key for key, _count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]]

    return {
        "frequently_valid_categories": frequent_keys(category_counts["valid"]),
        "frequently_invalid_categories": frequent_keys(category_counts["invalid"]),
        "frequently_valid_files": frequent_keys(file_counts["valid"]),
        "frequently_invalid_files": frequent_keys(file_counts["invalid"]),
    }


def compact_review_state(state: dict | None) -> dict:
    if not isinstance(state, dict):
        return {}
    return {
        "pr_summary": state.get("pr_summary", ""),
        "ranked_findings": state.get("ranked_findings", []),
        "merge_decision": state.get("merge_decision", {}),
        "review_budget": state.get("pr_context", {}).get("review_budget", state.get("review_budget", {})),
        "review_model_policy": state.get("pr_context", {}).get("review_model_policy", state.get("review_model_policy", {})),
        "deterministic_issues": state.get("deterministic_issues", []),
        "dropped_findings": state.get("dropped_findings", []),
        "grounding_summary": state.get("grounding_summary", {}),
        "finding_feedback": state.get("finding_feedback", {}),
        "feedback_summary": state.get("feedback_summary", {}),
        "final_review_posted": state.get("final_review_posted"),
        "rerun_comparison": state.get("rerun_comparison", {}),
    }


def as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def require_dashboard_auth(x_dashboard_key: str | None) -> None:
    # Keep dashboard/review actions private when key is configured.
    if DASHBOARD_API_KEY and x_dashboard_key != DASHBOARD_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def checkpoint_to_dict(snapshot) -> dict:
    config = getattr(snapshot, "config", {}) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    checkpoint_id = configurable.get("checkpoint_id")
    values = getattr(snapshot, "values", {}) or {}
    metadata = getattr(snapshot, "metadata", {}) or {}
    next_nodes = list(getattr(snapshot, "next", ()) or ())
    step = metadata.get("step") or metadata.get("source") or "checkpoint"
    return {
        "checkpoint_id": checkpoint_id,
        "step": step,
        "summary": f"{step}: next {', '.join(next_nodes) if next_nodes else 'END'}",
        "next_nodes": next_nodes,
        "metadata": metadata,
        "state": compact_review_state(values if isinstance(values, dict) else {}),
    }


async def process_opened_pr(metadata: dict) -> None:
    review_id = None
    try:
        pool = app.state.db
        graph = app.state.graph
        if metadata.get("draft") and metadata.get("action") != "ready_for_review":
            pr_context = build_queued_pr_context(metadata)
            existing_review = await get_existing_review_for_event(pool, pr_context)
            if existing_review is not None:
                return
            review_id = await save_review(pool, pr_context, ReviewStatus.QUEUED.value)
            queued_result = {
                "merge_decision": {
                    "decision": "queued",
                    "reason": "Draft PR queued until it is marked ready for review.",
                },
                "ranked_findings": [],
            }
            await update_review(pool, review_id, ReviewStatus.QUEUED.value, queued_result)
            await post_pr_comment(
                metadata["repository"],
                metadata["pr_number"],
                "🤖 Automated review queued because this pull request is still a draft. Analysis will run when it is marked ready for review.",
            )
            return

        metadata["human_feedback_memory"] = await summarize_human_feedback(
            pool,
            metadata["repository"],
            metadata.get("author"),
        )
        raw_diff = await get_pr_diff(
            metadata["repository"],
            metadata["pr_number"],
        )
        review_rules = await get_repo_review_rules(
            metadata["repository"],
            metadata.get("target_sha") or metadata.get("target_branch"),
        )
        pr_context = build_pr_context(metadata, raw_diff, review_rules)
        existing_review = await get_existing_review_for_event(pool, pr_context)
        if existing_review is not None:
            logger.info(
                "Skipping duplicate webhook",
                extra={
                    "repo": pr_context["repository"],
                    "pr_number": pr_context["pr_number"],
                    "source_sha": pr_context.get("source_sha"),
                    "existing_review_id": existing_review["id"],
                },
            )
            return
        review_id = await save_review(pool, pr_context, ReviewStatus.PENDING.value)

        # Post a quick status comment while analysis continues.
        comment = (
            f"🤖 **Automated Review Started**\n\n"
            f"Reviewing **{pr_context['total_files_changed']} file(s)** across "
            f"`{metadata['source_branch']}` → `{metadata['target_branch']}`.\n\n"
            f"Review budget truncated input: **{pr_context['review_budget']['truncated']}**.\n\n"
            f"Full review coming shortly."
        )
        await post_pr_comment(
            metadata["repository"],
            metadata["pr_number"],
            comment,
        )

        if graph is None:
            final_result = {
                "merge_decision": {
                    "decision": "needs_review",
                    "reason": "Review agent unavailable because required dependencies are missing.",
                },
                "ranked_findings": [],
            }
            await update_review(pool, review_id, ReviewStatus.FAILED.value, final_result)
            await post_pr_comment(
                metadata["repository"],
                metadata["pr_number"],
                "⚠️ Review agent unavailable because required dependencies are missing.",
            )
        else:
            config = {"configurable": {"thread_id": str(review_id)}}
            graph_state = await asyncio.to_thread(graph.invoke, {"pr_context": pr_context}, config)
            await update_review(
                pool,
                review_id,
                ReviewStatus.AWAITING_APPROVAL.value,
                compact_review_state(graph_state),
            )
            await post_pr_comment(
                metadata["repository"],
                metadata["pr_number"],
                (
                    "🤖 Analysis complete and paused before final merge decision.\n\n"
                    f"Review ID: `{review_id}`\n"
                    "Use the review API to inspect findings and approve to resume."
                ),
            )
    except Exception as e:
        # Keep webhook responder fast; report processing failures asynchronously.
        if review_id is not None:
            error_payload = {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "metadata": {
                    "repository": metadata.get("repository"),
                    "pr_number": metadata.get("pr_number"),
                    "action": metadata.get("action"),
                    "source_sha": metadata.get("source_sha"),
                    "delivery_id": metadata.get("delivery_id"),
                },
            }
            await update_review(
                app.state.db,
                review_id,
                ReviewStatus.FAILED.value,
                error_payload,
            )
        await post_pr_comment(
            metadata["repository"],
            metadata["pr_number"],
            f"⚠️ Automated review failed: `{str(e)}`",
        )


# -----------------------------
# GitHub webhook
# -----------------------------
@app.on_event("startup")
async def startup() -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required")
    app.state.db = await asyncpg.create_pool(DATABASE_URL)
    checkpointer = None
    store = None

    try:
        checkpoint_module = importlib.import_module("langgraph.checkpoint.postgres")
        postgres_saver_cls = getattr(checkpoint_module, "PostgresSaver")
        checkpointer = postgres_saver_cls.from_conn_string(DATABASE_URL)
        checkpointer.setup()
    except ModuleNotFoundError:
        logger.warning("LangGraph Postgres checkpointer is unavailable; starting without checkpoint persistence.")

    try:
        store_module = importlib.import_module("langgraph.store.postgres")
        postgres_store_cls = getattr(store_module, "PostgresStore")
        store = postgres_store_cls.from_conn_string(DATABASE_URL)
        store.setup()
    except ModuleNotFoundError:
        logger.warning("LangGraph Postgres store is unavailable; starting without review memory persistence.")

    app.state.checkpointer = checkpointer
    app.state.store = store
    app.state.graph = build_graph(checkpointer=checkpointer, store=store)


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.db.close()
    checkpointer = getattr(app.state, "checkpointer", None)
    if checkpointer is not None:
        checkpointer.close()
    store = getattr(app.state, "store", None)
    if store is not None and hasattr(store, "close"):
        store.close()


@app.get("/reviews/{id}")
async def get_review_findings(
    id: int, x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key")
):
    require_dashboard_auth(x_dashboard_key)
    row = await get_review(app.state.db, id)
    if row is None:
        raise HTTPException(status_code=404, detail="Review not found")

    result = as_dict(row["result"])
    findings = result.get("ranked_findings", [])

    return {
        "id": row["id"],
        "repo": row["repo"],
        "pr_number": row["pr_number"],
        "status": row["status"],
        "title": as_dict(row["pr_context"]).get("title", ""),
        "author": as_dict(row["pr_context"]).get("author", ""),
        "findings": findings,
        "timeline": review_timeline(row, result),
        "feedback_summary": feedback_summary(result),
        "current_state": result,
    }


@app.get("/reviews")
async def list_reviews(
    repo: str | None = Query(default=None),
    pr_number: int | None = Query(default=None),
    status: str | None = Query(default=None),
    author: str | None = Query(default=None),
    decision: str | None = Query(default=None),
    search: str | None = Query(default=None),
    x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key"),
):
    require_dashboard_auth(x_dashboard_key)
    rows = await list_review_rows(
        app.state.db,
        repo=repo,
        pr_number=pr_number,
        status=status,
        author=author,
        decision=decision,
        search=search,
    )
    reviews = []
    for row in rows:
        result = as_dict(row["result"])
        pr_context = as_dict(row["pr_context"])
        reviews.append(
            {
                "id": row["id"],
                "repo": row["repo"],
                "pr_number": row["pr_number"],
                "status": row["status"],
                "decision": result.get("merge_decision", {}).get("decision"),
                "title": pr_context.get("title", ""),
                "author": pr_context.get("author", ""),
                "draft": pr_context.get("draft", False),
                "findings_count": len(result.get("ranked_findings", [])),
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                "error": result.get("error"),
                "error_type": result.get("error_type"),
            }
        )
    return reviews


@app.get("/reviews/{id}/history")
async def get_review_history(
    id: int, x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key")
):
    require_dashboard_auth(x_dashboard_key)
    row = await get_review(app.state.db, id)
    if row is None:
        raise HTTPException(status_code=404, detail="Review not found")
    if app.state.graph is None:
        raise HTTPException(status_code=503, detail="Review graph unavailable")

    config = {"configurable": {"thread_id": str(id)}}
    snapshots = await asyncio.to_thread(lambda: list(app.state.graph.get_state_history(config)))
    return {
        "id": id,
        "history": [checkpoint_to_dict(snapshot) for snapshot in snapshots],
    }


@app.post("/reviews/{id}/rerun")
async def rerun_from_checkpoint(
    id: int,
    payload: dict,
    x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key"),
):
    require_dashboard_auth(x_dashboard_key)
    row = await get_review(app.state.db, id)
    if row is None:
        raise HTTPException(status_code=404, detail="Review not found")
    if app.state.graph is None:
        raise HTTPException(status_code=503, detail="Review graph unavailable")

    checkpoint_id = payload.get("checkpoint_id")
    if not checkpoint_id:
        raise HTTPException(status_code=400, detail="Missing checkpoint_id")

    config = {
        "configurable": {
            "thread_id": str(id),
            "checkpoint_id": str(checkpoint_id),
        }
    }
    rerun_state = await asyncio.to_thread(app.state.graph.invoke, None, config)
    compact_rerun_state = compact_review_state(rerun_state)
    previous_state = as_dict(row["result"])
    compact_rerun_state["rerun_comparison"] = compare_findings(
        previous_state.get("ranked_findings", []),
        compact_rerun_state.get("ranked_findings", []),
    )
    await update_review(app.state.db, id, ReviewStatus.AWAITING_APPROVAL.value, compact_rerun_state)
    return {
        "ok": True,
        "id": id,
        "status": ReviewStatus.AWAITING_APPROVAL.value,
        "comparison": compact_rerun_state["rerun_comparison"],
        "result": compact_rerun_state,
    }


@app.post("/reviews/{id}/post-final-review")
async def post_final_review(
    id: int,
    payload: dict | None = None,
    x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key"),
):
    require_dashboard_auth(x_dashboard_key)
    row = await get_review(app.state.db, id)
    if row is None:
        raise HTTPException(status_code=404, detail="Review not found")

    result = as_dict(row["result"])
    if not result:
        raise HTTPException(status_code=409, detail="Review has no result to post")
    if row["status"] in {ReviewStatus.QUEUED.value, ReviewStatus.FAILED.value}:
        raise HTTPException(status_code=409, detail=f"Cannot post final review while status is {row['status']}")

    payload = payload or {}
    include_inline = bool(payload.get("inline", True))
    pr_context = as_dict(row["pr_context"])
    repo = pr_context.get("repository", row["repo"])
    pr_number = pr_context.get("pr_number", row["pr_number"])
    await post_pr_comment(repo, pr_number, format_review_comment(result))
    inline_summary = {"posted": 0, "skipped": 0, "errors": []}
    if include_inline:
        inline_summary = await post_inline_review_comments(
            repo,
            pr_number,
            pr_context,
            result.get("ranked_findings", []),
        )

    posted_payload = {
        "posted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": "Summary comment posted" + (" with inline findings." if include_inline else "."),
        "inline_comments": inline_summary,
    }
    updated_result = {
        **result,
        "final_review_posted": posted_payload,
    }
    await update_review_result(app.state.db, id, updated_result)
    return {"ok": True, "id": id, "final_review_posted": posted_payload}


@app.post("/reviews/{id}/approve")
async def approve_review(
    id: int, x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key")
):
    require_dashboard_auth(x_dashboard_key)
    row = await get_review(app.state.db, id)
    if row is None:
        raise HTTPException(status_code=404, detail="Review not found")
    if row["status"] != ReviewStatus.AWAITING_APPROVAL.value:
        raise HTTPException(status_code=409, detail="Review is not awaiting approval")
    if app.state.graph is None:
        raise HTTPException(status_code=503, detail="Review graph unavailable")

    config = {"configurable": {"thread_id": str(id)}}
    final_result = await asyncio.to_thread(app.state.graph.invoke, None, config)
    compact_final_result = compact_review_state(final_result)
    await update_review(app.state.db, id, ReviewStatus.COMPLETED.value, compact_final_result)

    pr_context = as_dict(row["pr_context"])
    await post_pr_comment(
        pr_context.get("repository", row["repo"]),
        pr_context.get("pr_number", row["pr_number"]),
        format_review_comment(final_result),
    )
    inline_summary = await post_inline_review_comments(
        pr_context.get("repository", row["repo"]),
        pr_context.get("pr_number", row["pr_number"]),
        pr_context,
        compact_final_result.get("ranked_findings", []),
    )
    compact_final_result["final_review_posted"] = {
        "posted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": "Summary comment posted after approval.",
        "inline_comments": inline_summary,
    }
    await update_review_result(app.state.db, id, compact_final_result)
    return {"ok": True, "id": id, "status": ReviewStatus.COMPLETED.value, "result": compact_final_result}


@app.post("/reviews/{id}/findings/{finding_index}/feedback")
async def mark_finding_feedback(
    id: int,
    finding_index: int,
    payload: dict,
    x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key"),
):
    require_dashboard_auth(x_dashboard_key)
    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {"valid", "invalid"}:
        raise HTTPException(status_code=400, detail="verdict must be valid or invalid")
    row = await get_review(app.state.db, id)
    if row is None:
        raise HTTPException(status_code=404, detail="Review not found")

    result = as_dict(row["result"])
    findings = result.get("ranked_findings", [])
    if finding_index < 0 or finding_index >= len(findings):
        raise HTTPException(status_code=404, detail="Finding not found")

    feedback = dict(result.get("finding_feedback", {}))
    entry = {
        "verdict": verdict,
        "note": str(payload.get("note", "")).strip(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    feedback[str(finding_index)] = entry
    findings[finding_index] = {
        **findings[finding_index],
        "human_feedback": entry,
    }
    updated_result = {
        **result,
        "ranked_findings": findings,
        "finding_feedback": feedback,
    }
    updated_result["feedback_summary"] = feedback_summary(updated_result)
    await update_review_result(app.state.db, id, updated_result)
    return {"ok": True, "id": id, "finding_index": finding_index, "feedback": entry, "summary": updated_result["feedback_summary"]}


@app.post("/reviews/{id}/reject")
async def reject_review(
    id: int,
    payload: dict | None = None,
    x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key"),
):
    require_dashboard_auth(x_dashboard_key)
    row = await get_review(app.state.db, id)
    if row is None:
        raise HTTPException(status_code=404, detail="Review not found")
    if row["status"] != ReviewStatus.AWAITING_APPROVAL.value:
        raise HTTPException(status_code=409, detail="Review is not awaiting approval")

    reason = "Rejected by human reviewer."
    if payload and payload.get("reason"):
        reason = str(payload["reason"]).strip() or reason

    previous_state = as_dict(row["result"])
    rejected_result = {
        **previous_state,
        "merge_decision": {
            "decision": "reject",
            "reason": reason,
            "source": "human",
        },
        "human_decision": {
            "decision": "reject",
            "reason": reason,
        },
    }

    await update_review(app.state.db, id, ReviewStatus.COMPLETED.value, rejected_result)

    pr_context = as_dict(row["pr_context"])
    await post_pr_comment(
        pr_context.get("repository", row["repo"]),
        pr_context.get("pr_number", row["pr_number"]),
        f"⛔ Human review decision: **reject**\n\nReason: {reason}",
    )
    return {"ok": True, "id": id, "status": ReviewStatus.COMPLETED.value, "result": rejected_result}


@app.post("/reviews/{id}/request-changes")
async def request_changes_review(
    id: int,
    payload: dict | None = None,
    x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key"),
):
    require_dashboard_auth(x_dashboard_key)
    row = await get_review(app.state.db, id)
    if row is None:
        raise HTTPException(status_code=404, detail="Review not found")
    if row["status"] != ReviewStatus.AWAITING_APPROVAL.value:
        raise HTTPException(status_code=409, detail="Review is not awaiting approval")

    reason = "Changes requested by human reviewer."
    if payload and payload.get("reason"):
        reason = str(payload["reason"]).strip() or reason

    previous_state = as_dict(row["result"])
    requested_changes_result = {
        **previous_state,
        "merge_decision": {
            "decision": "needs_review",
            "reason": reason,
            "source": "human",
        },
        "human_decision": {
            "decision": "request_changes",
            "reason": reason,
        },
    }

    await update_review(app.state.db, id, ReviewStatus.COMPLETED.value, requested_changes_result)

    pr_context = as_dict(row["pr_context"])
    await post_pr_comment(
        pr_context.get("repository", row["repo"]),
        pr_context.get("pr_number", row["pr_number"]),
        f"📝 Human review decision: **request changes**\n\nReason: {reason}",
    )
    return {
        "ok": True,
        "id": id,
        "status": ReviewStatus.COMPLETED.value,
        "result": requested_changes_result,
    }


@app.get("/memory")
async def repository_memory(
    repo: str | None = Query(default=None),
    x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key"),
):
    require_dashboard_auth(x_dashboard_key)
    rows = await list_review_rows(app.state.db, repo=repo)
    risky_files: dict[str, int] = {}
    authors: dict[str, int] = {}
    categories: dict[str, int] = {}
    decisions: dict[str, int] = {}

    for row in rows:
        result = as_dict(row["result"])
        pr_context = as_dict(row["pr_context"])
        author = pr_context.get("author", "unknown")
        authors[author] = authors.get(author, 0) + 1
        decision = result.get("merge_decision", {}).get("decision", "unknown")
        decisions[decision] = decisions.get(decision, 0) + 1
        for finding in result.get("ranked_findings", []):
            if not isinstance(finding, dict):
                continue
            if finding.get("severity", "").lower() in {"critical", "high"} and finding.get("file"):
                risky_files[finding["file"]] = risky_files.get(finding["file"], 0) + 1
            category = finding.get("category", "general")
            categories[category] = categories.get(category, 0) + 1

    def ranked_counts(values: dict[str, int]) -> list[dict]:
        return [
            {"name": key, "count": count}
            for key, count in sorted(values.items(), key=lambda item: item[1], reverse=True)[:25]
        ]

    return {
        "repo": repo,
        "reviews_count": len(rows),
        "recurring_risky_files": ranked_counts(risky_files),
        "authors": ranked_counts(authors),
        "categories": ranked_counts(categories),
        "decisions": ranked_counts(decisions),
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key")):
    require_dashboard_auth(x_dashboard_key)
    with open("dashboard.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/readyz")
async def readyz():
    if not getattr(app.state, "db", None):
        raise HTTPException(status_code=503, detail="DB pool not initialized")
    try:
        await app.state.db.fetchval("SELECT 1")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB not ready: {exc}") from exc
    return {"ok": True}


@app.post("/github/webhook")
async def github_webhook(
    request: Request,
    x_github_event: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
    x_github_delivery: str | None = Header(default=None),
):
    raw_body = await request.body()

    # Distinguish missing header from invalid signature
    if x_hub_signature_256 is None:
        raise HTTPException(status_code=400, detail="Missing X-Hub-Signature-256 header")

    if not verify_github_signature(raw_body, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid GitHub signature")

    # Parse from already-read bytes to avoid consuming the stream twice
    payload = json.loads(raw_body)

    # Only newly opened PRs
    if x_github_event == "pull_request" and payload["action"] in {
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
    }:
        metadata = extract_pr_metadata(payload)
        metadata["delivery_id"] = x_github_delivery
        asyncio.create_task(process_opened_pr(metadata))

    return {"ok": True}
