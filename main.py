import hmac
import hashlib
import json
import os
import asyncio
import contextlib
import importlib
import logging
import socket
import traceback
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any
import httpx
import asyncpg
from agent import (
    DEFAULT_STRONG_REVIEW_MODEL,
    DEFAULT_TRIAGE_MODEL,
    build_graph,
    finding_fingerprint,
    merge_decision as calculate_merge_decision,
    merge_llm_usage_into_budget,
    reset_llm_usage,
    summarize_llm_usage,
)
from evaluation import evaluation_provenance, production_metrics
from fastapi import FastAPI, Request, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from observability import (
    log_event,
    metrics_response,
    record_finding_feedback_metric,
    record_review_evaluation,
    record_review_outcome,
    register_request_logging_middleware,
    setup_logging,
)
from review_queue import (
    attach_review_to_job,
    enqueue_review_job,
    review_worker_loop,
    run_migrations,
)
from review_intelligence import IntelligenceConfig, build_repository_intelligence

setup_logging()


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    await startup()
    try:
        yield
    finally:
        await shutdown()


app = FastAPI(lifespan=lifespan)
register_request_logging_middleware(app)
logger = logging.getLogger("pr_review")

GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
DASHBOARD_API_KEY = os.environ.get("DASHBOARD_API_KEY")
HTTP_TIMEOUT_SECONDS = float(os.environ.get("HTTP_TIMEOUT_SECONDS", "20"))
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development").strip().lower()
GITHUB_CHECK_RUN_NAME = os.environ.get("GITHUB_CHECK_RUN_NAME", "PR Reviewer")
REVIEW_WORKER_ENABLED = os.environ.get("REVIEW_WORKER_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
REVIEW_JOB_MAX_ATTEMPTS = int(os.environ.get("REVIEW_JOB_MAX_ATTEMPTS", "5"))
REVIEW_JOB_LEASE_SECONDS = int(os.environ.get("REVIEW_JOB_LEASE_SECONDS", "180"))
REVIEW_JOB_POLL_SECONDS = float(os.environ.get("REVIEW_JOB_POLL_SECONDS", "1"))
REVIEW_JOB_BACKOFF_SECONDS = float(os.environ.get("REVIEW_JOB_BACKOFF_SECONDS", "5"))
REVIEW_JOB_MAX_BACKOFF_SECONDS = float(os.environ.get("REVIEW_JOB_MAX_BACKOFF_SECONDS", "300"))
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
INTELLIGENCE_CONFIG = IntelligenceConfig.from_environment()


def is_production() -> bool:
    return ENVIRONMENT in {"production", "prod"}


def dashboard_auth_required() -> bool:
    if is_production():
        return True
    return os.environ.get("REQUIRE_DASHBOARD_AUTH", "").strip().lower() in {"1", "true", "yes"}


def validate_runtime_settings() -> None:
    if REVIEW_JOB_MAX_ATTEMPTS < 1:
        raise RuntimeError("REVIEW_JOB_MAX_ATTEMPTS must be at least 1")
    if REVIEW_JOB_LEASE_SECONDS < 3:
        raise RuntimeError("REVIEW_JOB_LEASE_SECONDS must be at least 3")
    if REVIEW_JOB_POLL_SECONDS <= 0:
        raise RuntimeError("REVIEW_JOB_POLL_SECONDS must be greater than 0")
    if REVIEW_JOB_BACKOFF_SECONDS <= 0 or REVIEW_JOB_MAX_BACKOFF_SECONDS <= 0:
        raise RuntimeError("Review job backoff settings must be greater than 0")


class ReviewStatus(str, Enum):
    PENDING = "pending"
    RETRYING = "retrying"
    QUEUED = "queued"
    AWAITING_APPROVAL = "awaiting_approval"
    FINALIZING = "finalizing"
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


async def get_live_pr_metadata(repo: str, pr_number: int) -> dict[str, Any]:
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=503, detail="GitHub token is not configured")
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    async def _request():
        timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.get(url, headers=headers)

    try:
        response = await github_request_with_retry(_request)
        response.raise_for_status()
        return extract_pr_metadata({
            "action": "synchronize",
            "repository": {"full_name": repo},
            "pull_request": response.json(),
        })
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to load the latest pull request head: {exc.response.status_code}",
        ) from exc


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
        "triage_model": str(controls.get("triage_model") or os.environ.get("OPENAI_TRIAGE_MODEL", DEFAULT_TRIAGE_MODEL)),
        "strong_model": str(controls.get("strong_model") or os.environ.get("OPENAI_STRONG_REVIEW_MODEL", os.environ.get("OPENAI_REVIEW_MODEL", DEFAULT_STRONG_REVIEW_MODEL))),
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
            "triage_model": os.environ.get("OPENAI_TRIAGE_MODEL", DEFAULT_TRIAGE_MODEL),
            "strong_model": os.environ.get("OPENAI_STRONG_REVIEW_MODEL", os.environ.get("OPENAI_REVIEW_MODEL", DEFAULT_STRONG_REVIEW_MODEL)),
            "selected_model": os.environ.get("OPENAI_TRIAGE_MODEL", DEFAULT_TRIAGE_MODEL),
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


async def post_status_comment(repo: str, pr_number: int, body: str) -> None:
    """Post a non-critical progress comment without failing the durable review job."""
    try:
        await post_pr_comment(repo, pr_number, body)
    except Exception as exc:
        logger.warning(
            "Unable to post non-critical PR status comment: %s",
            exc,
            extra={"repo": repo, "pr_number": pr_number},
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


def merge_decision_to_check_conclusion(decision: str) -> str:
    mapping = {
        "approve": "success",
        "reject": "failure",
        "needs_review": "neutral",
        "queued": "neutral",
    }
    return mapping.get(decision, "neutral")


def check_run_output_from_result(result: dict) -> dict[str, str]:
    decision = result.get("merge_decision", {}).get("decision", "needs_review")
    reason = result.get("merge_decision", {}).get("reason", "No reason provided.")
    summary = format_review_comment(result)
    if result.get("error"):
        summary = f"{summary}\n\n**Error:** {result['error']}"
    return {
        "title": f"PR review: {decision}",
        "summary": summary,
        "text": reason,
    }


async def create_github_check_run(
    repo: str,
    head_sha: str,
    review_id: int,
    *,
    status: str = "in_progress",
    conclusion: str | None = None,
    output: dict[str, str] | None = None,
) -> int | None:
    if not GITHUB_TOKEN or not head_sha:
        return None
    url = f"https://api.github.com/repos/{repo}/check-runs"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    payload: dict[str, Any] = {
        "name": GITHUB_CHECK_RUN_NAME,
        "head_sha": head_sha,
        "status": status,
        "external_id": f"pr-review-{review_id}",
    }
    if output:
        payload["output"] = output
    if status == "completed":
        payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload["conclusion"] = conclusion or "neutral"

    async def _request():
        timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(url, headers=headers, json=payload)

    try:
        response = await github_request_with_retry(_request)
        response.raise_for_status()
        check_run_id = response.json().get("id")
        log_event(
            logging.INFO,
            "github_check_run_created",
            "GitHub check run created",
            repo=repo,
            review_id=review_id,
            source_sha=head_sha,
            status=status,
        )
        return int(check_run_id) if check_run_id is not None else None
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Failed creating GitHub check run",
            extra={
                "event": "github_check_run_error",
                "repo": repo,
                "review_id": review_id,
                "status_code": exc.response.status_code,
            },
        )
        return None


async def update_github_check_run(
    repo: str,
    check_run_id: int,
    *,
    status: str = "completed",
    conclusion: str = "neutral",
    output: dict[str, str] | None = None,
) -> None:
    if not GITHUB_TOKEN:
        return
    url = f"https://api.github.com/repos/{repo}/check-runs/{check_run_id}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    payload: dict[str, Any] = {
        "status": status,
        "conclusion": conclusion,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if output:
        payload["output"] = output

    async def _request():
        timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.patch(url, headers=headers, json=payload)

    try:
        response = await github_request_with_retry(_request)
        response.raise_for_status()
        log_event(
            logging.INFO,
            "github_check_run_updated",
            "GitHub check run updated",
            repo=repo,
            status=status,
        )
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Failed updating GitHub check run",
            extra={
                "event": "github_check_run_error",
                "repo": repo,
                "status_code": exc.response.status_code,
            },
        )


async def sync_github_check_run(
    repo: str,
    head_sha: str | None,
    review_id: int,
    result: dict,
    *,
    existing_check_run_id: int | None = None,
) -> int | None:
    if not head_sha:
        return existing_check_run_id
    decision = result.get("merge_decision", {}).get("decision", "needs_review")
    output = check_run_output_from_result(result)
    if existing_check_run_id is None:
        return await create_github_check_run(
            repo,
            head_sha,
            review_id,
            status="completed",
            conclusion=merge_decision_to_check_conclusion(decision),
            output=output,
        )
    await update_github_check_run(
        repo,
        existing_check_run_id,
        status="completed",
        conclusion=merge_decision_to_check_conclusion(decision),
        output=output,
    )
    return existing_check_run_id


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
    active_findings = [finding for finding in findings if not finding_is_dismissed(finding)]
    for finding in active_findings[:10]:
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
            f"confidence {finding.get('confidence', 'n/a')}, "
            f"lifecycle {finding.get('lifecycle', 'new')}):\n\n"
            f"{finding.get('description', 'No description')}\n\n"
            f"**Why this matters:** {finding.get('why_this_matters') or finding.get('impact') or 'This can affect production behavior.'}\n\n"
            f"**Evidence:** {finding.get('evidence', 'changed diff line')}"
        )
        suggested_patch = finding.get("suggested_patch")
        if isinstance(suggested_patch, str):
            suggested_patch = suggested_patch.strip("\n")
            if (
                suggested_patch
                and len(suggested_patch) <= 4000
                and len(suggested_patch.splitlines()) <= 20
                and "```" not in suggested_patch
            ):
                body += f"\n\n```suggestion\n{suggested_patch}\n```"
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


async def get_or_create_review_version(
    pool: asyncpg.Pool,
    pr_context: dict,
    status: str,
    delivery_id: str | None,
) -> tuple[asyncpg.Record, bool]:
    """Return the single review row for a PR commit and allocate its version atomically."""
    source_sha = pr_context.get("source_sha")
    if not source_sha:
        raise ValueError("source_sha is required for a versioned review")

    lock_key = f"{pr_context['repository']}#{pr_context['pr_number']}"
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                lock_key,
            )
            existing = await connection.fetchrow(
                """
                SELECT id, repo, pr_number, status, pr_context, result,
                       source_sha, review_version, webhook_delivery_id,
                       lock_version, created_at, updated_at
                FROM pr_reviews
                WHERE repo = $1 AND pr_number = $2 AND source_sha = $3
                """,
                pr_context["repository"],
                pr_context["pr_number"],
                source_sha,
            )
            if existing is not None:
                return existing, False

            review_version = await connection.fetchval(
                """
                SELECT COALESCE(MAX(review_version), 0) + 1
                FROM pr_reviews
                WHERE repo = $1 AND pr_number = $2
                """,
                pr_context["repository"],
                pr_context["pr_number"],
            )
            created = await connection.fetchrow(
                """
                INSERT INTO pr_reviews (
                    repo,
                    pr_number,
                    status,
                    pr_context,
                    source_sha,
                    review_version,
                    webhook_delivery_id
                )
                VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
                RETURNING id, repo, pr_number, status, pr_context, result,
                          source_sha, review_version, webhook_delivery_id,
                          lock_version, created_at, updated_at
                """,
                pr_context["repository"],
                pr_context["pr_number"],
                status,
                json.dumps(pr_context),
                source_sha,
                int(review_version),
                delivery_id,
            )
            return created, True


async def update_review_context(pool: asyncpg.Pool, review_id: int, pr_context: dict) -> None:
    await pool.execute(
        """
        UPDATE pr_reviews
        SET pr_context = $1::jsonb,
            lock_version = lock_version + 1,
            updated_at = NOW()
        WHERE id = $2
        """,
        json.dumps(pr_context),
        review_id,
    )


async def transition_review_status(
    pool: asyncpg.Pool,
    review_id: int,
    *,
    expected_statuses: list[str],
    new_status: str,
) -> asyncpg.Record | None:
    return await pool.fetchrow(
        """
        UPDATE pr_reviews
        SET status = $2,
            lock_version = lock_version + 1,
            updated_at = NOW()
        WHERE id = $1 AND status = ANY($3::text[])
        RETURNING id, repo, pr_number, status, pr_context, result,
                  source_sha, review_version, webhook_delivery_id,
                  lock_version, created_at, updated_at
        """,
        review_id,
        new_status,
        expected_statuses,
    )


async def update_review(
    pool: asyncpg.Pool, review_id: int, status: str, result: dict
) -> None:
    await pool.execute(
        """
        UPDATE pr_reviews
        SET status = $1,
            result = $2,
            lock_version = lock_version + 1,
            updated_at = NOW()
        WHERE id = $3
        """,
        status,
        json.dumps(result),
        review_id,
    )


async def update_review_if_status(
    pool: asyncpg.Pool,
    review_id: int,
    *,
    expected_status: str,
    new_status: str,
    result: dict,
) -> asyncpg.Record | None:
    return await pool.fetchrow(
        """
        UPDATE pr_reviews
        SET status = $2,
            result = $3::jsonb,
            lock_version = lock_version + 1,
            updated_at = NOW()
        WHERE id = $1 AND status = $4
        RETURNING id, repo, pr_number, status, pr_context, result,
                  source_sha, review_version, webhook_delivery_id,
                  lock_version, created_at, updated_at
        """,
        review_id,
        new_status,
        json.dumps(result),
        expected_status,
    )


async def update_review_result(pool: asyncpg.Pool, review_id: int, result: dict) -> None:
    await pool.execute(
        """
        UPDATE pr_reviews
        SET result = $1,
            lock_version = lock_version + 1,
            updated_at = NOW()
        WHERE id = $2
        """,
        json.dumps(result),
        review_id,
    )


async def get_review(pool: asyncpg.Pool, review_id: int) -> asyncpg.Record | None:
    return await pool.fetchrow(
        """
        SELECT id, repo, pr_number, status, pr_context, result,
               source_sha, review_version, webhook_delivery_id,
               lock_version, created_at, updated_at
        FROM pr_reviews
        WHERE id = $1
        """,
        review_id,
    )


async def get_review_by_source(
    pool: asyncpg.Pool,
    repo: str,
    pr_number: int,
    source_sha: str,
) -> asyncpg.Record | None:
    return await pool.fetchrow(
        """
        SELECT id, repo, pr_number, status, pr_context, result,
               source_sha, review_version, webhook_delivery_id,
               lock_version, created_at, updated_at
        FROM pr_reviews
        WHERE repo = $1 AND pr_number = $2 AND source_sha = $3
        """,
        repo,
        pr_number,
        source_sha,
    )


async def get_previous_review_version(
    pool: asyncpg.Pool,
    repo: str,
    pr_number: int,
    review_version: int,
) -> asyncpg.Record | None:
    return await pool.fetchrow(
        """
        SELECT id, repo, pr_number, status, pr_context, result,
               source_sha, review_version, webhook_delivery_id,
               lock_version, created_at, updated_at
        FROM pr_reviews
        WHERE repo = $1
          AND pr_number = $2
          AND review_version < $3
          AND result IS NOT NULL
          AND result ? 'ranked_findings'
        ORDER BY review_version DESC
        LIMIT 1
        """,
        repo,
        pr_number,
        review_version,
    )


async def record_finding_feedback(
    pool: asyncpg.Pool,
    review_id: int,
    finding_index: int,
    entry: dict,
) -> dict | None:
    """Serialize validation and feedback mutation under a row lock."""
    async with pool.acquire() as connection:
        async with connection.transaction():
            row = await connection.fetchrow(
                """
                SELECT result, pr_context, repo, pr_number, review_version
                FROM pr_reviews
                WHERE id = $1
                FOR UPDATE
                """,
                review_id,
            )
            if row is None:
                return None

            result = as_dict(row["result"])
            findings = list(result.get("ranked_findings", []))
            if finding_index < 0 or finding_index >= len(findings):
                raise IndexError("Finding not found")

            feedback = dict(result.get("finding_feedback", {}))
            feedback[str(finding_index)] = entry
            existing_finding = findings[finding_index]
            lifecycle_before_dismissal = existing_finding.get("lifecycle_before_dismissal")
            if entry["verdict"] in {"invalid", "dismissed"} and existing_finding.get("lifecycle") != "dismissed":
                lifecycle_before_dismissal = existing_finding.get("lifecycle", "new")
            findings[finding_index] = {
                **existing_finding,
                "human_feedback": entry,
                "lifecycle_before_dismissal": lifecycle_before_dismissal,
            }
            updated_result = {
                **result,
                "ranked_findings": findings,
                "finding_feedback": feedback,
            }
            previous_row = await get_previous_review_version(
                connection,
                row["repo"],
                int(row["pr_number"]),
                int(row["review_version"]),
            )
            previous_result = as_dict(previous_row["result"]) if previous_row else {}
            updated_result = enrich_developer_experience(
                updated_result,
                as_dict(row["pr_context"]),
                previous_result,
                dict(previous_row) if previous_row else None,
            )
            updated_result["feedback_summary"] = feedback_summary(updated_result)

            await connection.execute(
                """
                INSERT INTO review_finding_feedback (
                    review_id, finding_index, verdict, note, updated_at
                )
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (review_id, finding_index)
                DO UPDATE SET verdict = EXCLUDED.verdict,
                              note = EXCLUDED.note,
                              updated_at = NOW()
                """,
                review_id,
                finding_index,
                entry["verdict"],
                entry.get("note", ""),
            )
            await connection.execute(
                """
                UPDATE pr_reviews
                SET result = $1::jsonb,
                    lock_version = lock_version + 1,
                    updated_at = NOW()
                WHERE id = $2
                """,
                json.dumps(updated_result),
                review_id,
            )
            return updated_result


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
        SELECT id, repo, pr_number, status, result, pr_context,
               source_sha, review_version, webhook_delivery_id,
               lock_version, created_at, updated_at
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
    all_findings = result.get("ranked_findings", [])
    findings = [finding for finding in all_findings if not finding_is_dismissed(finding)][:5]
    budget = result.get("review_budget", {})
    grounding = result.get("grounding_summary", {})
    intelligence = result.get("review_intelligence", {}) or {}
    test_suggestions = result.get("test_suggestions", [])[:5]
    baseline = result.get("risk_baseline", {}) or {}
    lifecycle = (result.get("finding_lifecycle", {}) or {}).get("counts", {})
    manager_summary = result.get("manager_summary", {}) or {}

    analysis_errors = result.get("analysis_errors", [])
    lines = [
        "## 🤖 Automated PR Review",
        f"**Decision:** `{decision}`",
        f"**Reason:** {reason}",
        "",
        f"**Grounding:** {grounding.get('verified', 0)} verified, {grounding.get('dropped', 0)} dropped as ungrounded.",
    ]
    execution = intelligence.get("execution", {}) or {}
    if intelligence:
        tool_commands = execution.get("commands", [])
        passed = sum(1 for command in tool_commands if command.get("exit_code") == 0)
        lines.append(
            f"**Repository intelligence:** `{intelligence.get('status', 'unknown')}`; "
            f"{passed}/{len(tool_commands)} analysis commands passed; "
            f"sandbox `{(intelligence.get('isolation') or {}).get('backend', 'unknown')}`."
        )
    if baseline:
        lines.append(f"**Risk baseline:** {baseline.get('statement', 'Unavailable')}")
    if lifecycle:
        lines.append(
            "**Finding lifecycle:** "
            + ", ".join(f"{name} `{lifecycle.get(name, 0)}`" for name in ("new", "recurring", "fixed", "dismissed"))
        )
    if manager_summary:
        tests = manager_summary.get("tests", {})
        lines.append(
            f"**Manager view:** {len(manager_summary.get('top_risks', []))} priority risk(s); "
            f"tests `{tests.get('status', 'not_run')}`."
        )
    if analysis_errors:
        failed_nodes = ", ".join(
            sorted({str(err.get("node", "unknown")) for err in analysis_errors if isinstance(err, dict)})
        )
        lines.append(f"**Analysis errors:** automated checks failed for `{failed_nodes}`.")

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
        if budget.get("llm_total_tokens"):
            lines.append(
                f"**LLM usage:** {budget.get('llm_total_tokens', 0)} tokens "
                f"({budget.get('llm_prompt_tokens', 0)} prompt / "
                f"{budget.get('llm_completion_tokens', 0)} completion); "
                f"estimated cost `${budget.get('llm_estimated_cost_usd', 0.0):.4f}`."
            )

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
            lifecycle_status = finding.get("lifecycle", "new")
            line = finding.get("line")
            location = f"{file_name}:{line}" if line else file_name
            lines.append(f"- [{sev}/{cat}/{finding_type}/{lifecycle_status}/confidence={confidence}] {desc} (`{location}`)")
            if finding.get("why_this_matters"):
                lines.append(f"  - Why this matters: {finding.get('why_this_matters')}")
            if finding.get("evidence"):
                lines.append(f"  - Evidence: {finding.get('evidence')}")
            if finding.get("evidence_sources"):
                lines.append(
                    f"  - Verification: `{finding.get('verification_level', 'diff_grounded')}` via "
                    + ", ".join(f"`{source}`" for source in finding.get("evidence_sources", []))
                )
            context = finding.get("line_context", [])[:3]
            for item in context:
                lines.append(
                    f"  - `{item.get('change', 'changed')}:{item.get('line')}` {item.get('content', '')[:160]}"
                )

    if test_suggestions:
        lines.extend(["", "### Suggested Tests"])
        for suggestion in test_suggestions:
            lines.append(
                f"- {suggestion.get('title', 'Add regression coverage')} "
                f"(`{suggestion.get('file', 'unknown')}`): {suggestion.get('reason', '')}"
            )

    heatmap = result.get("file_risk_heatmap", [])[:5]
    if heatmap:
        lines.extend(["", "### File Risk Heatmap"])
        for item in heatmap:
            lines.append(
                f"- `{item.get('level', 'none')}` **{item.get('file', 'unknown')}** — "
                f"score {item.get('score', 0)}, {item.get('findings', 0)} finding(s)"
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


RISK_WEIGHTS = {"critical": 10.0, "high": 6.0, "medium": 3.0, "low": 1.0}


def finding_is_dismissed(finding: dict) -> bool:
    verdict = (finding.get("human_feedback") or {}).get("verdict")
    return finding.get("lifecycle") == "dismissed" or verdict in {"invalid", "dismissed"}


def finding_risk_score(finding: dict) -> float:
    if finding_is_dismissed(finding):
        return 0.0
    severity = str(finding.get("effective_severity") or finding.get("severity") or "low").lower()
    raw_confidence = finding.get("confidence")
    confidence = 0.5 if raw_confidence is None else float(raw_confidence)
    return RISK_WEIGHTS.get(severity, 1.0) * max(0.0, min(confidence, 1.0))


def total_risk_score(findings: list[dict]) -> float:
    return round(sum(finding_risk_score(finding) for finding in findings if isinstance(finding, dict)), 2)


def file_risk_heatmap(pr_context: dict, findings: list[dict]) -> list[dict]:
    files = {
        str(file.get("filename")): {
            "file": str(file.get("filename")),
            "score": 0.0,
            "findings": 0,
            "categories": set(),
            "highest_severity": None,
        }
        for file in pr_context.get("files", [])
        if file.get("filename")
    }
    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    for finding in findings:
        if not isinstance(finding, dict) or finding_is_dismissed(finding):
            continue
        file_name = str(finding.get("file", ""))
        entry = files.setdefault(file_name, {
            "file": file_name,
            "score": 0.0,
            "findings": 0,
            "categories": set(),
            "highest_severity": None,
        })
        entry["score"] += finding_risk_score(finding)
        entry["findings"] += 1
        entry["categories"].add(str(finding.get("category", "general")))
        severity = str(finding.get("effective_severity") or finding.get("severity") or "low").lower()
        current = entry["highest_severity"]
        if current is None or severity_rank.get(severity, 0) > severity_rank.get(current, 0):
            entry["highest_severity"] = severity

    heatmap = []
    for entry in files.values():
        score = round(entry["score"], 2)
        if score >= 10:
            level = "critical"
        elif score >= 6:
            level = "high"
        elif score >= 3:
            level = "medium"
        elif score > 0:
            level = "low"
        else:
            level = "none"
        heatmap.append({
            **entry,
            "score": score,
            "level": level,
            "categories": sorted(entry["categories"]),
        })
    return sorted(heatmap, key=lambda item: (-item["score"], item["file"]))


def risk_baseline(
    previous_findings: list[dict],
    current_findings: list[dict],
    *,
    has_previous: bool | None = None,
) -> dict:
    previous = total_risk_score(previous_findings)
    current = total_risk_score(current_findings)
    delta = round(current - previous, 2)
    if has_previous is None:
        has_previous = bool(previous_findings)
    if not has_previous:
        direction = "baseline"
        percent_change = None
        statement = f"Initial risk baseline is {current:.2f} points."
    else:
        direction = "improved" if delta < 0 else "worsened" if delta > 0 else "unchanged"
        percent_change = round((delta / previous) * 100, 1) if previous else None
        change = abs(delta)
        percent_text = f" ({abs(percent_change):.1f}%)" if percent_change is not None else ""
        statement = f"Risk {direction} by {change:.2f} points{percent_text}."
    return {
        "previous_score": previous,
        "current_score": current,
        "delta": delta,
        "percent_change": percent_change,
        "direction": direction,
        "statement": statement,
    }


def refresh_finding_lifecycle(result: dict) -> dict:
    findings = []
    counts = {"new": 0, "recurring": 0, "fixed": 0, "dismissed": 0}
    lifecycle = dict(result.get("finding_lifecycle", {}))
    fixed = list(lifecycle.get("fixed", []))
    counts["fixed"] = len(fixed)
    for finding in result.get("ranked_findings", []):
        item = dict(finding)
        verdict = (item.get("human_feedback") or {}).get("verdict")
        if verdict in {"invalid", "dismissed"}:
            item["lifecycle"] = "dismissed"
        elif verdict == "valid" and item.get("lifecycle") == "dismissed":
            item["lifecycle"] = item.get("lifecycle_before_dismissal") or "new"
        status = item.get("lifecycle", "new")
        if status not in counts:
            status = "new"
            item["lifecycle"] = status
        counts[status] += 1
        findings.append(item)
    return {
        **result,
        "ranked_findings": findings,
        "finding_lifecycle": {**lifecycle, "counts": counts, "fixed": fixed},
    }


def apply_persisted_finding_feedback(
    result: dict,
    persisted_result: dict,
    *,
    recompute_decision: bool = False,
) -> dict:
    persisted_by_fingerprint = {
        finding_fingerprint(finding): finding
        for finding in persisted_result.get("ranked_findings", [])
        if isinstance(finding, dict)
    }
    findings = []
    for finding in result.get("ranked_findings", []):
        persisted = persisted_by_fingerprint.get(finding_fingerprint(finding), {})
        human_feedback = persisted.get("human_feedback")
        item = dict(finding)
        if human_feedback:
            item["human_feedback"] = human_feedback
            if persisted.get("lifecycle_before_dismissal"):
                item["lifecycle_before_dismissal"] = persisted["lifecycle_before_dismissal"]
        if persisted.get("lifecycle") == "dismissed" or (
            human_feedback or {}
        ).get("verdict") in {"invalid", "dismissed"}:
            item["lifecycle"] = "dismissed"
        findings.append(item)
    merged = refresh_finding_lifecycle({
        **result,
        "ranked_findings": findings,
        "finding_feedback": persisted_result.get("finding_feedback", result.get("finding_feedback", {})),
    })
    merged["feedback_summary"] = feedback_summary(merged)
    if recompute_decision:
        merged.update(calculate_merge_decision(merged))
    return merged


def enrich_developer_experience(
    result: dict,
    pr_context: dict,
    previous_result: dict | None = None,
    previous_review: dict | None = None,
) -> dict:
    enriched = refresh_finding_lifecycle(result)
    findings = enriched.get("ranked_findings", [])
    previous_findings = (previous_result or {}).get("ranked_findings", [])
    heatmap = file_risk_heatmap(pr_context, findings)
    baseline = risk_baseline(
        previous_findings,
        findings,
        has_previous=previous_review is not None,
    )
    failed_finding_nodes = {
        "logic_issues",
        "security_issues",
        "performance_issues",
        "contract_issues",
        "test_evaluation",
    }
    baseline_reliable = not any(
        isinstance(error, dict) and error.get("node") in failed_finding_nodes
        for error in enriched.get("analysis_errors", [])
    )
    baseline["reliable"] = baseline_reliable
    if not baseline_reliable:
        baseline["direction"] = "unavailable"
        baseline["statement"] = "Risk comparison is unavailable because one or more finding analyzers failed."
    if previous_review:
        baseline["previous_review_version"] = previous_review.get("review_version")
        baseline["previous_source_sha"] = previous_review.get("source_sha")
    active = [finding for finding in findings if not finding_is_dismissed(finding)]
    top_risks = [
        {
            "severity": finding.get("effective_severity") or finding.get("severity"),
            "category": finding.get("category", "general"),
            "file": finding.get("file"),
            "description": finding.get("description"),
        }
        for finding in sorted(active, key=finding_risk_score, reverse=True)[:3]
    ]
    tests = ((enriched.get("review_intelligence") or {}).get("execution") or {}).get("tests", {})
    manager_summary = {
        "headline": baseline["statement"],
        "decision": (enriched.get("merge_decision") or {}).get("decision", "pending"),
        "scope": {
            "changed_files": len(pr_context.get("files", [])),
            "impacted_files": len(
                ((enriched.get("review_intelligence") or {}).get("review_scope") or {}).get("impacted_files", [])
            ),
        },
        "lifecycle": enriched.get("finding_lifecycle", {}).get("counts", {}),
        "top_risks": top_risks,
        "highest_risk_files": heatmap[:5],
        "tests": {
            "status": "not_run" if tests.get("exit_code") is None else "passed" if tests.get("exit_code") == 0 else "failed",
            "exit_code": tests.get("exit_code"),
        },
        "engineering_summary": enriched.get("pr_summary", ""),
    }
    return {
        **enriched,
        "file_risk_heatmap": heatmap,
        "risk_baseline": baseline,
        "manager_summary": manager_summary,
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
    invalid = sum(1 for item in feedback.values() if item.get("verdict") in {"invalid", "dismissed"})
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
            if verdict == "dismissed":
                verdict = "invalid"
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


def compact_review_state(state: dict | None, *, llm_usage: dict | None = None) -> dict:
    if not isinstance(state, dict):
        return {}
    review_budget = state.get("pr_context", {}).get("review_budget", state.get("review_budget", {}))
    if llm_usage:
        review_budget = merge_llm_usage_into_budget(review_budget, llm_usage)
    return {
        "pr_summary": state.get("pr_summary", ""),
        "ranked_findings": state.get("ranked_findings", []),
        "merge_decision": state.get("merge_decision", {}),
        "review_budget": review_budget,
        "review_model_policy": state.get("pr_context", {}).get("review_model_policy", state.get("review_model_policy", {})),
        "deterministic_issues": state.get("deterministic_issues", []),
        "static_issues": state.get("static_issues", []),
        "test_suggestions": state.get("test_suggestions", []),
        "review_intelligence": state.get("pr_context", {}).get("intelligence", {}),
        "dropped_findings": state.get("dropped_findings", []),
        "grounding_summary": state.get("grounding_summary", {}),
        "evidence_summary": state.get("evidence_summary", {}),
        "finding_lifecycle": state.get("finding_lifecycle", {}),
        "finding_feedback": state.get("finding_feedback", {}),
        "feedback_summary": state.get("feedback_summary", {}),
        "final_review_posted": state.get("final_review_posted"),
        "rerun_comparison": state.get("rerun_comparison", {}),
        "github_check_run_id": state.get("github_check_run_id"),
        "analysis_errors": state.get("analysis_errors", []),
    }


def attach_evaluation_metrics(
    result: dict,
    *,
    started_at: float,
    model_policy: dict | None = None,
) -> dict:
    enriched = dict(result)
    budget = enriched.get("review_budget", {}) or {}
    duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
    estimated_cost_usd = round(float(budget.get("llm_estimated_cost_usd", 0.0) or 0.0), 6)
    enriched["evaluation_metrics"] = {
        "time_to_review_ms": duration_ms,
        "estimated_cost_usd": estimated_cost_usd,
        **evaluation_provenance(model_policy),
    }
    return enriched


def finalize_graph_result(graph_state: dict | None) -> dict:
    usage = summarize_llm_usage()
    compact = compact_review_state(graph_state, llm_usage=usage if usage.get("calls") else None)
    return compact


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
    if dashboard_auth_required():
        if not DASHBOARD_API_KEY or x_dashboard_key != DASHBOARD_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return
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


async def process_opened_pr(
    metadata: dict,
    *,
    job: dict[str, Any] | None = None,
) -> None:
    review_started_at = time.perf_counter()
    review_id = None
    check_run_id = None
    repo = metadata.get("repository", "")
    pr_number = metadata.get("pr_number")
    try:
        pool = app.state.db
        graph = app.state.graph
        initial_context = build_queued_pr_context(metadata)
        is_waiting_draft = metadata.get("draft") and metadata.get("action") != "ready_for_review"
        initial_status = ReviewStatus.QUEUED.value if is_waiting_draft else ReviewStatus.PENDING.value
        review_row, created = await get_or_create_review_version(
            pool,
            initial_context,
            initial_status,
            metadata.get("delivery_id"),
        )
        review_id = int(review_row["id"])
        if job is not None:
            job["review_id"] = review_id
            await attach_review_to_job(pool, int(job["id"]), review_id)

        if is_waiting_draft:
            queued_result = {
                "merge_decision": {
                    "decision": "queued",
                    "reason": "Draft PR queued until it is marked ready for review.",
                },
                "ranked_findings": [],
            }
            if not created:
                if not as_dict(review_row["result"]):
                    await update_review(pool, review_id, ReviewStatus.QUEUED.value, queued_result)
                return
            check_run_id = await create_github_check_run(
                repo,
                metadata.get("source_sha", ""),
                review_id,
                status="completed",
                conclusion="neutral",
                output=check_run_output_from_result(queued_result),
            )
            if check_run_id is not None:
                queued_result["github_check_run_id"] = check_run_id
            await update_review(pool, review_id, ReviewStatus.QUEUED.value, queued_result)
            await post_status_comment(
                metadata["repository"],
                metadata["pr_number"],
                "🤖 Automated review queued because this pull request is still a draft. Analysis will run when it is marked ready for review.",
            )
            record_review_outcome("queued", repo=repo, pr_number=pr_number, review_id=review_id)
            return

        if not created:
            current_status = str(review_row["status"])
            is_retry = bool(job and int(job.get("attempts", 1)) > 1)
            can_resume_draft = (
                metadata.get("action") == "ready_for_review"
                and current_status == ReviewStatus.QUEUED.value
            )
            can_retry = is_retry and current_status in {
                ReviewStatus.PENDING.value,
                ReviewStatus.RETRYING.value,
                ReviewStatus.FAILED.value,
            }
            if not can_resume_draft and not can_retry:
                logger.info(
                    "Skipping review event for an existing PR version",
                    extra={
                        "repo": repo,
                        "pr_number": pr_number,
                        "source_sha": metadata.get("source_sha"),
                        "review_id": review_id,
                        "status": current_status,
                    },
                )
                return
            claimed = await transition_review_status(
                pool,
                review_id,
                expected_statuses=[current_status],
                new_status=ReviewStatus.PENDING.value,
            )
            if claimed is None:
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
        await update_review_context(pool, review_id, pr_context)
        check_run_id = await create_github_check_run(
            repo,
            pr_context.get("source_sha", ""),
            review_id,
            status="in_progress",
            output={
                "title": "PR review in progress",
                "summary": "Automated review started.",
            },
        )

        if created or metadata.get("action") == "ready_for_review":
            comment = (
                f"🤖 **Automated Review Started**\n\n"
                f"Reviewing **{pr_context['total_files_changed']} file(s)** across "
                f"`{metadata['source_branch']}` → `{metadata['target_branch']}`.\n\n"
                f"Review version: **{review_row['review_version']}**.\n"
                f"Review budget truncated input: **{pr_context['review_budget']['truncated']}**.\n\n"
                f"Full review coming shortly."
            )
            await post_status_comment(
                metadata["repository"],
                metadata["pr_number"],
                comment,
            )

        try:
            intelligence = await asyncio.to_thread(
                build_repository_intelligence,
                metadata,
                pr_context,
                github_token=GITHUB_TOKEN,
                config=INTELLIGENCE_CONFIG,
            )
        except Exception as exc:
            if is_production():
                raise
            logger.exception("Repository intelligence failed: %s", exc)
            intelligence = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc)[:500],
            }
        pr_context["intelligence"] = intelligence
        previous_review_row = await get_previous_review_version(
            pool,
            repo,
            int(pr_number),
            int(review_row["review_version"]),
        )
        previous_result = as_dict(previous_review_row["result"]) if previous_review_row else {}
        pr_context["previous_findings"] = previous_result.get("ranked_findings", [])
        await update_review_context(pool, review_id, pr_context)

        if graph is None:
            final_result = {
                "merge_decision": {
                    "decision": "needs_review",
                    "reason": "Review agent unavailable because required dependencies are missing.",
                },
                "ranked_findings": [],
            }
            final_result = enrich_developer_experience(
                final_result,
                pr_context,
                previous_result,
                dict(previous_review_row) if previous_review_row else None,
            )
            final_result = attach_evaluation_metrics(
                final_result,
                started_at=review_started_at,
                model_policy=pr_context.get("review_model_policy"),
            )
            record_review_evaluation(
                final_result["evaluation_metrics"]["time_to_review_ms"],
                final_result["evaluation_metrics"]["estimated_cost_usd"],
                repo=repo,
                pr_number=pr_number,
                review_id=review_id,
            )
            if check_run_id is not None:
                final_result["github_check_run_id"] = check_run_id
            await sync_github_check_run(
                repo,
                pr_context.get("source_sha"),
                review_id,
                final_result,
                existing_check_run_id=check_run_id,
            )
            await update_review(pool, review_id, ReviewStatus.FAILED.value, final_result)
            await post_status_comment(
                metadata["repository"],
                metadata["pr_number"],
                "⚠️ Review agent unavailable because required dependencies are missing.",
            )
            record_review_outcome(
                "failed",
                repo=repo,
                pr_number=pr_number,
                review_id=review_id,
                error_type="graph_unavailable",
            )
        else:
            config = {"configurable": {"thread_id": str(review_id)}}
            reset_llm_usage()
            graph_state = await asyncio.to_thread(graph.invoke, {"pr_context": pr_context}, config)
            compact_state = finalize_graph_result(graph_state)
            compact_state = enrich_developer_experience(
                compact_state,
                pr_context,
                previous_result,
                dict(previous_review_row) if previous_review_row else None,
            )
            compact_state = attach_evaluation_metrics(
                compact_state,
                started_at=review_started_at,
                model_policy=pr_context.get("review_model_policy"),
            )
            record_review_evaluation(
                compact_state["evaluation_metrics"]["time_to_review_ms"],
                compact_state["evaluation_metrics"]["estimated_cost_usd"],
                repo=repo,
                pr_number=pr_number,
                review_id=review_id,
            )
            if check_run_id is not None:
                compact_state["github_check_run_id"] = check_run_id
            await sync_github_check_run(
                repo,
                pr_context.get("source_sha"),
                review_id,
                compact_state,
                existing_check_run_id=check_run_id,
            )
            await update_review(
                pool,
                review_id,
                ReviewStatus.AWAITING_APPROVAL.value,
                compact_state,
            )
            await post_status_comment(
                metadata["repository"],
                metadata["pr_number"],
                (
                    "🤖 Analysis complete and paused before final merge decision.\n\n"
                    f"Review ID: `{review_id}`\n"
                    "Use the review API to inspect findings and approve to resume."
                ),
            )
            record_review_outcome(
                "awaiting_approval",
                repo=repo,
                pr_number=pr_number,
                review_id=review_id,
            )
    except Exception as e:
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
                "merge_decision": {
                    "decision": "needs_review",
                    "reason": "Automated review failed and will require retry or human review.",
                },
            }
            error_payload = attach_evaluation_metrics(
                error_payload,
                started_at=review_started_at,
            )
            record_review_evaluation(
                error_payload["evaluation_metrics"]["time_to_review_ms"],
                error_payload["evaluation_metrics"]["estimated_cost_usd"],
                repo=repo,
                pr_number=pr_number,
                review_id=review_id,
            )
            if check_run_id is not None:
                error_payload["github_check_run_id"] = check_run_id
            await sync_github_check_run(
                repo,
                metadata.get("source_sha"),
                review_id,
                error_payload,
                existing_check_run_id=check_run_id,
            )
            await update_review(
                app.state.db,
                review_id,
                ReviewStatus.FAILED.value,
                error_payload,
            )
        record_review_outcome(
            "failed",
            repo=repo,
            pr_number=pr_number,
            review_id=review_id,
            error_type=type(e).__name__,
        )
        raise


async def process_review_job(job: dict[str, Any]) -> None:
    metadata = as_dict(job.get("payload"))
    if not metadata:
        raise ValueError("Review job payload is invalid")
    await process_opened_pr(metadata, job=job)


async def handle_review_job_failure(
    job: dict[str, Any],
    error: Exception,
    terminal: bool,
    retry_delay: float,
) -> None:
    review_id = job.get("review_id")
    if review_id is None:
        return
    row = await get_review(app.state.db, int(review_id))
    if row is None:
        return
    result = as_dict(row["result"])
    result["job_failure"] = {
        "attempt": int(job.get("attempts", 1)),
        "max_attempts": int(job.get("max_attempts", REVIEW_JOB_MAX_ATTEMPTS)),
        "terminal": terminal,
        "retry_in_seconds": retry_delay,
        "error_type": type(error).__name__,
        "message": str(error)[:500],
    }
    status = ReviewStatus.FAILED.value if terminal else ReviewStatus.RETRYING.value
    await update_review(app.state.db, int(review_id), status, result)


# -----------------------------
# LangGraph Postgres persistence
# -----------------------------
def create_langgraph_pg_pool(database_url: str):
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    return ConnectionPool(
        database_url,
        min_size=1,
        max_size=10,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )


def init_langgraph_persistence(database_url: str) -> tuple[Any, Any, Any]:
    """Create checkpointer and store backed by a shared psycopg connection pool.

    PostgresSaver/PostgresStore.from_conn_string() return context managers for
    short-lived scripts. Long-running servers must construct them with a pool.
    """
    checkpoint_module = importlib.import_module("langgraph.checkpoint.postgres")
    store_module = importlib.import_module("langgraph.store.postgres")
    postgres_saver_cls = getattr(checkpoint_module, "PostgresSaver")
    postgres_store_cls = getattr(store_module, "PostgresStore")

    pool = create_langgraph_pg_pool(database_url)
    try:
        checkpointer = postgres_saver_cls(pool)
        checkpointer.setup()
        store = postgres_store_cls(pool)
        store.setup()
    except Exception:
        pool.close()
        raise
    return checkpointer, store, pool


# -----------------------------
# GitHub webhook
# -----------------------------
async def startup() -> None:
    validate_runtime_settings()
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required")
    if dashboard_auth_required() and not DASHBOARD_API_KEY:
        raise RuntimeError("DASHBOARD_API_KEY is required in production")
    app.state.db = await asyncpg.create_pool(DATABASE_URL)
    try:
        await run_migrations(app.state.db, MIGRATIONS_DIR)
    except Exception as exc:
        await app.state.db.close()
        raise RuntimeError("Database migrations failed during startup") from exc
    checkpointer = None
    store = None
    langgraph_pool = None

    try:
        checkpointer, store, langgraph_pool = init_langgraph_persistence(DATABASE_URL)
    except ModuleNotFoundError as exc:
        if is_production():
            await app.state.db.close()
            raise RuntimeError(
                "LangGraph PostgreSQL persistence is required in production"
            ) from exc
        logger.warning(
            "LangGraph Postgres persistence is unavailable; starting without checkpoint/memory persistence."
        )
    except Exception as exc:
        if is_production():
            await app.state.db.close()
            raise RuntimeError(
                "LangGraph PostgreSQL persistence failed during production startup"
            ) from exc
        logger.warning("Failed initializing LangGraph Postgres persistence: %s", exc)

    INTELLIGENCE_CONFIG.validate(
        production=is_production(),
        worker_enabled=REVIEW_WORKER_ENABLED,
    )

    app.state.checkpointer = checkpointer
    app.state.store = store
    app.state.langgraph_pool = langgraph_pool
    app.state.graph = build_graph(checkpointer=checkpointer, store=store)
    if is_production() and app.state.graph is None:
        if langgraph_pool is not None:
            langgraph_pool.close()
        await app.state.db.close()
        raise RuntimeError("Review graph is required in production")

    app.state.worker_stop = asyncio.Event()
    app.state.worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    app.state.review_worker_task = None
    if REVIEW_WORKER_ENABLED:
        app.state.review_worker_task = asyncio.create_task(
            review_worker_loop(
                app.state.db,
                worker_id=app.state.worker_id,
                handler=process_review_job,
                stop_event=app.state.worker_stop,
                lease_seconds=REVIEW_JOB_LEASE_SECONDS,
                poll_seconds=REVIEW_JOB_POLL_SECONDS,
                base_backoff_seconds=REVIEW_JOB_BACKOFF_SECONDS,
                max_backoff_seconds=REVIEW_JOB_MAX_BACKOFF_SECONDS,
                failure_handler=handle_review_job_failure,
            ),
            name="durable-pr-review-worker",
        )


async def shutdown() -> None:
    worker_stop = getattr(app.state, "worker_stop", None)
    if worker_stop is not None:
        worker_stop.set()
    worker_task = getattr(app.state, "review_worker_task", None)
    if worker_task is not None:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
    langgraph_pool = getattr(app.state, "langgraph_pool", None)
    if langgraph_pool is not None:
        langgraph_pool.close()
    await app.state.db.close()


async def load_recent_evaluation_results(pool: asyncpg.Pool, limit: int = 1000) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT result
        FROM pr_reviews
        WHERE result IS NOT NULL
        ORDER BY id DESC
        LIMIT $1
        """,
        limit,
    )
    return [as_dict(row["result"]) for row in rows if as_dict(row["result"])]


@app.get("/evaluation/metrics")
async def get_evaluation_metrics(
    x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key"),
):
    require_dashboard_auth(x_dashboard_key)
    results = await load_recent_evaluation_results(app.state.db)
    return {
        **production_metrics(results),
        "provenance": evaluation_provenance(),
        "note": "Production precision/recall requires benchmark ground truth; use benchmarks/evaluate.py.",
    }


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
        "review_version": row["review_version"],
        "source_sha": row["source_sha"],
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
                "review_version": row["review_version"],
                "source_sha": row["source_sha"],
                "status": row["status"],
                "decision": result.get("merge_decision", {}).get("decision"),
                "title": pr_context.get("title", ""),
                "author": pr_context.get("author", ""),
                "draft": pr_context.get("draft", False),
                "findings_count": sum(
                    1 for finding in result.get("ranked_findings", [])
                    if not finding_is_dismissed(finding)
                ),
                "risk_direction": (result.get("risk_baseline") or {}).get("direction"),
                "risk_delta": (result.get("risk_baseline") or {}).get("delta"),
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


@app.get("/reviews/{id}/manager-summary")
async def get_manager_summary(
    id: int,
    x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key"),
):
    require_dashboard_auth(x_dashboard_key)
    row = await get_review(app.state.db, id)
    if row is None:
        raise HTTPException(status_code=404, detail="Review not found")
    result = as_dict(row["result"])
    return {
        "id": id,
        "repo": row["repo"],
        "pr_number": row["pr_number"],
        "review_version": row["review_version"],
        "summary": result.get("manager_summary", {}),
        "risk_baseline": result.get("risk_baseline", {}),
        "file_risk_heatmap": result.get("file_risk_heatmap", []),
        "finding_lifecycle": result.get("finding_lifecycle", {}),
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


    reset_llm_usage()
    rerun_state = await asyncio.to_thread(app.state.graph.invoke, None, config)
    compact_rerun_state = finalize_graph_result(rerun_state)
    previous_state = as_dict(row["result"])
    compact_rerun_state = apply_persisted_finding_feedback(
        compact_rerun_state,
        previous_state,
    )
    pr_context = as_dict(row["pr_context"])
    baseline_row = await get_previous_review_version(
        app.state.db,
        row["repo"],
        int(row["pr_number"]),
        int(row["review_version"]),
    )
    compact_rerun_state = enrich_developer_experience(
        compact_rerun_state,
        pr_context,
        as_dict(baseline_row["result"]) if baseline_row else {},
        dict(baseline_row) if baseline_row else None,
    )
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


@app.post("/reviews/{id}/rerun-latest")
async def rerun_after_new_commits(
    id: int,
    x_dashboard_key: str | None = Header(default=None, alias="X-Dashboard-Key"),
):
    require_dashboard_auth(x_dashboard_key)
    row = await get_review(app.state.db, id)
    if row is None:
        raise HTTPException(status_code=404, detail="Review not found")

    metadata = await get_live_pr_metadata(row["repo"], int(row["pr_number"]))
    latest_sha = metadata.get("source_sha")
    if not latest_sha or latest_sha == row["source_sha"]:
        raise HTTPException(status_code=409, detail="No new commits are available for review")

    existing = await get_review_by_source(
        app.state.db,
        row["repo"],
        int(row["pr_number"]),
        latest_sha,
    )
    if existing is not None:
        return {
            "ok": True,
            "created": False,
            "review_id": existing["id"],
            "status": existing["status"],
            "source_sha": latest_sha,
        }

    metadata["delivery_id"] = None
    job_id, created = await enqueue_review_job(
        app.state.db,
        idempotency_key=f"manual-rerun:{row['repo']}:{row['pr_number']}:{latest_sha}",
        metadata=metadata,
        max_attempts=REVIEW_JOB_MAX_ATTEMPTS,
    )
    return {
        "ok": True,
        "created": created,
        "job_id": job_id,
        "status": "queued",
        "source_sha": latest_sha,
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
    previous_post = result.get("final_review_posted") or {}
    if previous_post.get("status") == "posted" or previous_post.get("posted_at"):
        raise HTTPException(status_code=409, detail="Final review has already been posted")

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
        "status": "posted",
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
    if app.state.graph is None:
        raise HTTPException(status_code=503, detail="Review graph unavailable")
    row = await transition_review_status(
        app.state.db,
        id,
        expected_statuses=[ReviewStatus.AWAITING_APPROVAL.value],
        new_status=ReviewStatus.FINALIZING.value,
    )
    if row is None:
        existing = await get_review(app.state.db, id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Review not found")
        raise HTTPException(status_code=409, detail="Review is not awaiting approval")

    config = {"configurable": {"thread_id": str(id)}}
    try:
        reset_llm_usage()
        final_result = await asyncio.to_thread(app.state.graph.invoke, None, config)
        compact_final_result = finalize_graph_result(final_result)
    except Exception:
        await transition_review_status(
            app.state.db,
            id,
            expected_statuses=[ReviewStatus.FINALIZING.value],
            new_status=ReviewStatus.AWAITING_APPROVAL.value,
        )
        raise
    previous_result = as_dict(row["result"])
    compact_final_result = apply_persisted_finding_feedback(
        compact_final_result,
        previous_result,
        recompute_decision=True,
    )
    check_run_id = previous_result.get("github_check_run_id")
    if check_run_id is not None:
        compact_final_result["github_check_run_id"] = check_run_id
    pr_context = as_dict(row["pr_context"])
    baseline_row = await get_previous_review_version(
        app.state.db,
        row["repo"],
        int(row["pr_number"]),
        int(row["review_version"]),
    )
    compact_final_result = enrich_developer_experience(
        compact_final_result,
        pr_context,
        as_dict(baseline_row["result"]) if baseline_row else {},
        dict(baseline_row) if baseline_row else None,
    )
    if check_run_id is not None:
        compact_final_result["github_check_run_id"] = check_run_id
    try:
        await sync_github_check_run(
            pr_context.get("repository", row["repo"]),
            pr_context.get("source_sha"),
            id,
            compact_final_result,
            existing_check_run_id=check_run_id,
        )
        await update_review(app.state.db, id, ReviewStatus.COMPLETED.value, compact_final_result)
    except Exception:
        await transition_review_status(
            app.state.db,
            id,
            expected_statuses=[ReviewStatus.FINALIZING.value],
            new_status=ReviewStatus.AWAITING_APPROVAL.value,
        )
        raise

    try:
        await post_pr_comment(
            pr_context.get("repository", row["repo"]),
            pr_context.get("pr_number", row["pr_number"]),
            format_review_comment(compact_final_result),
        )
        inline_summary = await post_inline_review_comments(
            pr_context.get("repository", row["repo"]),
            pr_context.get("pr_number", row["pr_number"]),
            pr_context,
            compact_final_result.get("ranked_findings", []),
        )
        compact_final_result["final_review_posted"] = {
            "status": "posted",
            "posted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": "Summary comment posted after approval.",
            "inline_comments": inline_summary,
        }
    except Exception as exc:
        compact_final_result["final_review_posted"] = {
            "status": "failed",
            "posted_at": None,
            "summary": "Review completed, but publishing to GitHub failed. It can be retried.",
            "error": str(exc)[:500],
        }
    await update_review_result(app.state.db, id, compact_final_result)
    record_review_outcome(
        "completed",
        repo=pr_context.get("repository", row["repo"]),
        pr_number=pr_context.get("pr_number", row["pr_number"]),
        review_id=id,
    )
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
    if verdict not in {"valid", "invalid", "dismissed"}:
        raise HTTPException(status_code=400, detail="verdict must be valid, invalid, or dismissed")
    entry = {
        "verdict": verdict,
        "note": str(payload.get("note", "")).strip(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        updated_result = await record_finding_feedback(
            app.state.db,
            id,
            finding_index,
            entry,
        )
    except IndexError as exc:
        raise HTTPException(status_code=404, detail="Finding not found") from exc
    if updated_result is None:
        raise HTTPException(status_code=404, detail="Review not found")
    findings = updated_result.get("ranked_findings", [])
    finding = findings[finding_index] if 0 <= finding_index < len(findings) else {}
    record_finding_feedback_metric(
        verdict,
        str(finding.get("category", "general")),
        review_id=id,
    )
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

    pr_context = as_dict(row["pr_context"])
    check_run_id = previous_state.get("github_check_run_id")
    if check_run_id is not None:
        rejected_result["github_check_run_id"] = check_run_id
    updated = await update_review_if_status(
        app.state.db,
        id,
        expected_status=ReviewStatus.AWAITING_APPROVAL.value,
        new_status=ReviewStatus.COMPLETED.value,
        result=rejected_result,
    )
    if updated is None:
        raise HTTPException(status_code=409, detail="Review decision was already claimed")
    await sync_github_check_run(
        pr_context.get("repository", row["repo"]),
        pr_context.get("source_sha"),
        id,
        rejected_result,
        existing_check_run_id=check_run_id,
    )

    await post_status_comment(
        pr_context.get("repository", row["repo"]),
        pr_context.get("pr_number", row["pr_number"]),
        f"⛔ Human review decision: **reject**\n\nReason: {reason}",
    )
    record_review_outcome(
        "rejected",
        repo=pr_context.get("repository", row["repo"]),
        pr_number=pr_context.get("pr_number", row["pr_number"]),
        review_id=id,
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

    pr_context = as_dict(row["pr_context"])
    check_run_id = previous_state.get("github_check_run_id")
    if check_run_id is not None:
        requested_changes_result["github_check_run_id"] = check_run_id
    updated = await update_review_if_status(
        app.state.db,
        id,
        expected_status=ReviewStatus.AWAITING_APPROVAL.value,
        new_status=ReviewStatus.COMPLETED.value,
        result=requested_changes_result,
    )
    if updated is None:
        raise HTTPException(status_code=409, detail="Review decision was already claimed")
    await sync_github_check_run(
        pr_context.get("repository", row["repo"]),
        pr_context.get("source_sha"),
        id,
        requested_changes_result,
        existing_check_run_id=check_run_id,
    )

    await post_status_comment(
        pr_context.get("repository", row["repo"]),
        pr_context.get("pr_number", row["pr_number"]),
        f"📝 Human review decision: **request changes**\n\nReason: {reason}",
    )
    record_review_outcome(
        "request_changes",
        repo=pr_context.get("repository", row["repo"]),
        pr_number=pr_context.get("pr_number", row["pr_number"]),
        review_id=id,
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


@app.get("/metrics")
async def metrics():
    return metrics_response()


@app.get("/readyz")
async def readyz():
    if not getattr(app.state, "db", None):
        raise HTTPException(status_code=503, detail="DB pool not initialized")
    try:
        await app.state.db.fetchval("SELECT 1")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB not ready: {exc}") from exc
    worker_task = getattr(app.state, "review_worker_task", None)
    if REVIEW_WORKER_ENABLED and (worker_task is None or worker_task.done()):
        raise HTTPException(status_code=503, detail="Review worker is not running")
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

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    # Only newly opened PRs
    if x_github_event == "pull_request" and payload["action"] in {
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
    }:
        metadata = extract_pr_metadata(payload)
        metadata["delivery_id"] = x_github_delivery
        idempotency_key = (
            f"github:{x_github_delivery}"
            if x_github_delivery
            else f"payload:{hashlib.sha256(raw_body).hexdigest()}"
        )
        job_id, created = await enqueue_review_job(
            app.state.db,
            idempotency_key=idempotency_key,
            metadata=metadata,
            max_attempts=REVIEW_JOB_MAX_ATTEMPTS,
        )
        log_event(
            logging.INFO,
            "review_job_enqueued",
            "Review job durably enqueued" if created else "Duplicate review delivery ignored",
            repo=metadata["repository"],
            pr_number=metadata["pr_number"],
            delivery_id=x_github_delivery,
            review_id=job_id,
            status="queued" if created else "duplicate",
        )

    return {"ok": True}
