import hashlib
import hmac
import json

import asyncpg
import pytest
from fastapi.testclient import TestClient

import main
from tests.fakes import FakeGraph, FakePool


def sign_webhook(body: bytes, secret: str = "test_secret") -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


@pytest.fixture
def api_client(monkeypatch):
    fake_pool = FakePool()

    async def fake_create_pool(*_args, **_kwargs):
        return fake_pool

    def fake_import_module(name, package=None):
        if name.startswith("langgraph.checkpoint") or name.startswith("langgraph.store"):
            raise ModuleNotFoundError(name)
        return importlib.import_module(name, package)

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(main.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(main, "DASHBOARD_API_KEY", "test-dashboard-key")
    monkeypatch.setattr(main, "GITHUB_WEBHOOK_SECRET", "test_secret")
    monkeypatch.setattr(main, "GITHUB_TOKEN", "test_token")
    monkeypatch.setattr(main, "DATABASE_URL", "postgresql://test/test")

    async def noop_post(*_args, **_kwargs):
        return None

    async def noop_inline(*_args, **_kwargs):
        return {"posted": 0, "skipped": 0, "errors": []}

    monkeypatch.setattr(main, "post_pr_comment", noop_post)
    monkeypatch.setattr(main, "post_inline_review_comments", noop_inline)

    with TestClient(main.app) as client:
        yield client, fake_pool


def pull_request_payload(action: str = "opened", pr_number: int = 42) -> dict:
    return {
        "action": action,
        "repository": {"full_name": "org/repo"},
        "pull_request": {
            "number": pr_number,
            "title": "Test PR",
            "body": "Description",
            "user": {"login": "alice"},
            "state": "open",
            "html_url": f"https://example/pr/{pr_number}",
            "head": {"ref": "feature", "sha": "abc123"},
            "base": {"ref": "main", "sha": "def456"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "merged": False,
            "draft": False,
        },
    }


def test_healthz(api_client):
    client, _ = api_client
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_readyz(api_client):
    client, _ = api_client
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_github_webhook_valid_signature(api_client):
    client, _ = api_client
    body = json.dumps(pull_request_payload()).encode()
    response = client.post(
        "/github/webhook",
        content=body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": sign_webhook(body),
        },
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_github_webhook_invalid_signature(api_client):
    client, _ = api_client
    body = json.dumps(pull_request_payload()).encode()
    response = client.post(
        "/github/webhook",
        content=body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": "sha256=invalid",
        },
    )
    assert response.status_code == 401


def test_list_and_get_reviews_requires_auth(api_client):
    client, pool = api_client
    pool.seed_review(
        1,
        "org/repo",
        7,
        main.ReviewStatus.AWAITING_APPROVAL.value,
        {
            "repository": "org/repo",
            "pr_number": 7,
            "title": "Improve parser",
            "author": "alice",
        },
        {
            "ranked_findings": [
                {
                    "description": "Hardcoded secret",
                    "file": "app.py",
                    "line": 10,
                    "severity": "critical",
                    "category": "security",
                }
            ],
            "merge_decision": {"decision": "needs_review", "reason": "Paused"},
        },
    )

    unauthorized = client.get("/reviews")
    assert unauthorized.status_code == 401

    listed = client.get("/reviews", headers={"X-Dashboard-Key": "test-dashboard-key"})
    assert listed.status_code == 200
    data = listed.json()
    assert len(data) == 1
    assert data[0]["repo"] == "org/repo"
    assert data[0]["findings_count"] == 1

    detail = client.get("/reviews/1", headers={"X-Dashboard-Key": "test-dashboard-key"})
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] == main.ReviewStatus.AWAITING_APPROVAL.value
    assert len(body["findings"]) == 1


def test_reject_review(api_client):
    client, pool = api_client
    pool.seed_review(
        2,
        "org/repo",
        8,
        main.ReviewStatus.AWAITING_APPROVAL.value,
        {"repository": "org/repo", "pr_number": 8, "title": "Risky change", "author": "bob"},
        {"ranked_findings": [], "merge_decision": {"decision": "needs_review"}},
    )

    response = client.post(
        "/reviews/2/reject",
        headers={"X-Dashboard-Key": "test-dashboard-key"},
        json={"reason": "Security concerns"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == main.ReviewStatus.COMPLETED.value
    assert payload["result"]["merge_decision"]["decision"] == "reject"
    assert payload["result"]["human_decision"]["reason"] == "Security concerns"


def test_approve_review(api_client, monkeypatch):
    client, pool = api_client
    pool.seed_review(
        3,
        "org/repo",
        9,
        main.ReviewStatus.AWAITING_APPROVAL.value,
        {"repository": "org/repo", "pr_number": 9, "title": "Clean fix", "author": "carol"},
        {"ranked_findings": []},
    )
    main.app.state.graph = FakeGraph()

    response = client.post(
        "/reviews/3/approve",
        headers={"X-Dashboard-Key": "test-dashboard-key"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == main.ReviewStatus.COMPLETED.value
    assert payload["result"]["merge_decision"]["decision"] == "approve"
