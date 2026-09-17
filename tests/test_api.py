import hashlib
import hmac
import importlib
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
    monkeypatch.setattr(main, "REVIEW_WORKER_ENABLED", False)

    async def noop_migrations(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "run_migrations", noop_migrations)

    async def noop_post(*_args, **_kwargs):
        return None

    async def noop_inline(*_args, **_kwargs):
        return {"posted": 0, "skipped": 0, "errors": []}

    monkeypatch.setattr(main, "post_pr_comment", noop_post)
    monkeypatch.setattr(main, "post_inline_review_comments", noop_inline)

    async def noop_check_run(*_args, **_kwargs):
        return None

    async def noop_sync_check_run(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "create_github_check_run", noop_check_run)
    monkeypatch.setattr(main, "sync_github_check_run", noop_sync_check_run)
    monkeypatch.setattr(main, "ENVIRONMENT", "development")

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


def test_metrics_endpoint(api_client):
    client, _ = api_client
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "pr_review_http_requests_total" in response.text


def test_evaluation_metrics_endpoint(api_client):
    client, pool = api_client
    pool.seed_review(
        30,
        "org/repo",
        30,
        main.ReviewStatus.COMPLETED.value,
        {"repository": "org/repo", "pr_number": 30},
        {
            "evaluation_metrics": {"time_to_review_ms": 250, "estimated_cost_usd": 0.025},
            "ranked_findings": [{"category": "security"}],
            "finding_feedback": {"0": {"verdict": "valid"}},
        },
    )

    response = client.get(
        "/evaluation/metrics",
        headers={"X-Dashboard-Key": "test-dashboard-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["review_count"] == 1
    assert payload["time_to_review_ms"]["p50"] == 250.0
    assert payload["cost_per_pr_usd"]["mean"] == 0.025
    assert payload["human_acceptance_rate"] == 1.0
    assert len(payload["provenance"]["prompt_fingerprint"]) == 64


def test_production_startup_requires_dashboard_key(monkeypatch):
    fake_pool = FakePool()

    async def fake_create_pool(*_args, **_kwargs):
        return fake_pool

    def fake_import_module(name, package=None):
        if name.startswith("langgraph.checkpoint") or name.startswith("langgraph.store"):
            raise ModuleNotFoundError(name)
        return importlib.import_module(name, package)

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(main.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(main, "ENVIRONMENT", "production")
    monkeypatch.setattr(main, "DASHBOARD_API_KEY", None)
    monkeypatch.setattr(main, "DATABASE_URL", "postgresql://test/test")

    with pytest.raises(RuntimeError, match="DASHBOARD_API_KEY is required"):
        with TestClient(main.app):
            pass


def test_production_startup_requires_persistent_langgraph(monkeypatch):
    fake_pool = FakePool()

    async def fake_create_pool(*_args, **_kwargs):
        return fake_pool

    async def noop_migrations(*_args, **_kwargs):
        return None

    def fake_import_module(name, package=None):
        if name.startswith("langgraph.checkpoint") or name.startswith("langgraph.store"):
            raise ModuleNotFoundError(name)
        return importlib.import_module(name, package)

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(main, "run_migrations", noop_migrations)
    monkeypatch.setattr(main.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(main, "ENVIRONMENT", "production")
    monkeypatch.setattr(main, "DASHBOARD_API_KEY", "production-key")
    monkeypatch.setattr(main, "DATABASE_URL", "postgresql://test/test")

    with pytest.raises(RuntimeError, match="persistence is required"):
        with TestClient(main.app):
            pass


def test_github_webhook_valid_signature(api_client):
    client, pool = api_client
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
    assert len(pool.jobs) == 1


def test_github_webhook_delivery_is_idempotent(api_client):
    client, pool = api_client
    body = json.dumps(pull_request_payload()).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": "delivery-123",
        "X-Hub-Signature-256": sign_webhook(body),
    }
    first = client.post("/github/webhook", content=body, headers=headers)
    second = client.post("/github/webhook", content=body, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(pool.jobs) == 1


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

    duplicate = client.post(
        "/reviews/3/approve",
        headers={"X-Dashboard-Key": "test-dashboard-key"},
    )
    assert duplicate.status_code == 409


def test_finding_feedback_updates_atomically(api_client):
    client, pool = api_client
    pool.seed_review(
        4,
        "org/repo",
        10,
        main.ReviewStatus.AWAITING_APPROVAL.value,
        {"repository": "org/repo", "pr_number": 10, "title": "Feedback", "author": "dana"},
        {
            "ranked_findings": [
                {"description": "Possible issue", "file": "app.py", "line": 3}
            ]
        },
    )

    response = client.post(
        "/reviews/4/findings/0/feedback",
        headers={"X-Dashboard-Key": "test-dashboard-key"},
        json={"verdict": "valid", "note": "Confirmed locally"},
    )

    assert response.status_code == 200
    assert response.json()["summary"] == {"valid": 1, "invalid": 0, "total": 1}
    stored = pool.reviews[4]["result"]
    assert stored["finding_feedback"]["0"]["note"] == "Confirmed locally"
    assert stored["ranked_findings"][0]["human_feedback"]["verdict"] == "valid"


def test_dismissed_finding_updates_lifecycle(api_client):
    client, pool = api_client
    pool.seed_review(
        8,
        "org/repo",
        18,
        main.ReviewStatus.AWAITING_APPROVAL.value,
        {
            "repository": "org/repo",
            "pr_number": 18,
            "review_version": 1,
            "files": [{"filename": "app.py"}],
        },
        {"ranked_findings": [{"description": "Noise", "file": "app.py", "line": 2, "lifecycle": "new"}]},
    )

    response = client.post(
        "/reviews/8/findings/0/feedback",
        headers={"X-Dashboard-Key": "test-dashboard-key"},
        json={"verdict": "dismissed", "note": "Not applicable"},
    )

    assert response.status_code == 200
    assert response.json()["summary"] == {"valid": 0, "invalid": 1, "total": 1}
    assert pool.reviews[8]["result"]["ranked_findings"][0]["lifecycle"] == "dismissed"

    restored = client.post(
        "/reviews/8/findings/0/feedback",
        headers={"X-Dashboard-Key": "test-dashboard-key"},
        json={"verdict": "valid", "note": "Confirmed after review"},
    )
    assert restored.status_code == 200
    assert pool.reviews[8]["result"]["ranked_findings"][0]["lifecycle"] == "new"


def test_one_click_rerun_queues_latest_commit(api_client, monkeypatch):
    client, pool = api_client
    pool.seed_review(
        9,
        "org/repo",
        19,
        main.ReviewStatus.COMPLETED.value,
        {"repository": "org/repo", "pr_number": 19, "source_sha": "old-sha", "review_version": 1},
        {"ranked_findings": []},
    )

    async def fake_live_metadata(*_args, **_kwargs):
        return {
            "action": "synchronize",
            "repository": "org/repo",
            "pr_number": 19,
            "source_sha": "new-sha",
            "title": "Latest",
        }

    monkeypatch.setattr(main, "get_live_pr_metadata", fake_live_metadata)
    response = client.post(
        "/reviews/9/rerun-latest",
        headers={"X-Dashboard-Key": "test-dashboard-key"},
    )

    assert response.status_code == 200
    assert response.json()["created"] is True
    assert response.json()["source_sha"] == "new-sha"
    assert next(iter(pool.jobs.values()))["source_sha"] == "new-sha"


def test_manager_summary_endpoint(api_client):
    client, pool = api_client
    pool.seed_review(
        10,
        "org/repo",
        20,
        main.ReviewStatus.COMPLETED.value,
        {"repository": "org/repo", "pr_number": 20, "review_version": 2},
        {
            "manager_summary": {"headline": "Risk improved by 4 points."},
            "risk_baseline": {"direction": "improved", "delta": -4},
            "file_risk_heatmap": [{"file": "app.py", "score": 2}],
            "finding_lifecycle": {"counts": {"fixed": 1}},
        },
    )

    response = client.get(
        "/reviews/10/manager-summary",
        headers={"X-Dashboard-Key": "test-dashboard-key"},
    )

    assert response.status_code == 200
    assert response.json()["summary"]["headline"] == "Risk improved by 4 points."
    assert response.json()["risk_baseline"]["direction"] == "improved"
