import hashlib
import hmac
from datetime import datetime, timezone

import pytest

import agent
import main


@pytest.fixture(autouse=True)
def set_test_secrets(monkeypatch):
    monkeypatch.setattr(main, "GITHUB_WEBHOOK_SECRET", "test_secret")


def test_verify_github_signature_valid():
    raw_body = b'{"hello":"world"}'
    signature = "sha256=" + hmac.new(
        b"test_secret",
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    assert main.verify_github_signature(raw_body, signature) is True


def test_parse_diff_preserves_line_numbers():
    raw_diff = """diff --git a/foo.py b/foo.py
index 123..456 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 line1
-line2
+line2_changed
+line3_new
 line4
"""
    parsed = main.parse_diff(raw_diff)
    assert parsed[0]["filename"] == "foo.py"
    assert parsed[0]["added_line_details"][0]["line"] == 2
    assert parsed[0]["added_line_details"][1]["line"] == 3
    assert parsed[0]["removed_line_details"][0]["line"] == 2
    assert parsed[0]["removed_line_details"][0]["diff_position"] == 2
    assert parsed[0]["added_line_details"][0]["diff_position"] == 3


def test_prune_diff_limits_total_lines():
    parsed = [{
        "filename": "foo.py",
        "file_type": "python",
        "change_type": "modified",
        "added_lines": ["a"] * 500,
        "removed_lines": ["b"] * 500,
        "added_line_details": [{"line": i + 1, "content": "a"} for i in range(500)],
        "removed_line_details": [{"line": i + 1, "content": "b"} for i in range(500)],
        "chunks": [{"added": ["a"] * 500, "removed": ["b"] * 500}],
    }]
    pruned = main.prune_diff(parsed)
    assert len(pruned) == 1
    assert len(pruned[0]["added_lines"]) <= main.MAX_LINES_PER_FILE
    assert len(pruned[0]["removed_lines"]) <= main.MAX_LINES_PER_FILE


def test_build_pr_context_includes_budget_and_rules():
    raw_diff = """diff --git a/foo.py b/foo.py
index 123..456 100644
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-old
+new
"""
    metadata = {
        "pr_number": 1,
        "title": "t",
        "description": "d",
        "author": "alice",
        "action": "opened",
        "url": "https://example/pr/1",
        "source_branch": "feature",
        "source_sha": "abc",
        "target_branch": "main",
        "target_sha": "def",
        "repository": "org/repo",
    }
    context = main.build_pr_context(metadata, raw_diff, {
        "rules": ["check auth"],
        "cost_controls": {
            "max_files": 5,
            "prompt_char_budget": 5000,
            "triage_model": "cheap-model",
            "strong_model": "strong-model",
        },
    })
    assert context["source_sha"] == "abc"
    assert context["review_rules"]["rules"] == ["check auth"]
    assert context["review_budget"]["total_files_seen"] == 1
    assert context["review_budget"]["truncated"] is False
    assert context["review_budget"]["prompt_truncated"] is False
    assert context["review_budget"]["max_files"] == 5
    assert context["review_budget"]["cost_controls"]["triage_model"] == "cheap-model"
    assert context["review_model_policy"]["selected_model"] == "cheap-model"


def test_format_review_comment_contains_top_findings():
    comment = main.format_review_comment({
        "merge_decision": {"decision": "reject", "reason": "Critical issue"},
        "ranked_findings": [{
            "severity": "critical",
            "category": "security",
            "description": "Hardcoded secret",
            "file": "app.py",
            "line": 10,
            "confidence": 0.9,
            "finding_type": "definite_bug",
            "line_context": [{"line": 10, "change": "added", "content": "SECRET='x'"}],
        }],
        "grounding_summary": {"verified": 1, "dropped": 0},
    })
    assert "Automated PR Review" in comment
    assert "Hardcoded secret" in comment
    assert "confidence=0.9" in comment
    assert "app.py:10" in comment


def test_extract_pr_metadata():
    payload = {
        "action": "opened",
        "repository": {"full_name": "org/repo"},
        "pull_request": {
            "number": 7,
            "title": "Improve parser",
            "body": "desc",
            "user": {"login": "alice"},
            "state": "open",
            "html_url": "https://example/pr/7",
            "head": {"ref": "feature", "sha": "abc"},
            "base": {"ref": "main", "sha": "def"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "merged": False,
            "draft": True,
        },
    }
    metadata = main.extract_pr_metadata(payload)
    assert metadata["repository"] == "org/repo"
    assert metadata["pr_number"] == 7
    assert metadata["author"] == "alice"
    assert metadata["draft"] is True


def test_parse_review_rules_yaml():
    assert main.parse_review_rules("rules:\n  - check auth\n") == {"rules": ["check auth"]}


def test_structured_output_rejects_markdown_json():
    with pytest.raises(ValueError):
        agent.parse_structured_output(
            '```json\n{"findings":[]}\n```',
            agent.FindingResult,
        )


def test_rank_findings_deduplicates_and_verifier_adds_context():
    state = {
        "pr_context": {
            "files": [{
                "filename": "app.py",
                "added_line_details": [{"line": 5, "content": "return user.is_admin"}],
                "removed_line_details": [],
            }]
        },
        "logic_issues": [{
            "description": "Allows admin access without tenant check",
            "file": "app.py",
            "line": 5,
            "severity": "high",
            "confidence": 0.8,
            "finding_type": "definite_bug",
        }],
        "security_issues": [{
            "description": "Allows admin access without tenant check",
            "file": "app.py",
            "line": 5,
            "severity": "critical",
            "confidence": 0.9,
            "finding_type": "definite_bug",
        }],
        "deterministic_issues": [],
        "performance_issues": [],
        "contract_issues": [],
        "test_evaluation": [],
    }
    ranked = agent.rank_findings(state)["ranked_findings"]
    assert len(ranked) == 1
    assert ranked[0]["category"] == "logic/security"
    verified = agent.verify_findings_grounded({**state, "ranked_findings": ranked})
    assert verified["grounding_summary"] == {"verified": 1, "dropped": 0}
    assert verified["ranked_findings"][0]["line_context"][0]["line"] == 5
    assert verified["ranked_findings"][0]["evidence"] == "return user.is_admin"


def test_build_queued_pr_context_for_draft_pr():
    metadata = {
        "pr_number": 7,
        "title": "Draft parser",
        "description": "",
        "author": "alice",
        "action": "opened",
        "url": "https://example/pr/7",
        "source_branch": "feature",
        "source_sha": "abc",
        "target_branch": "main",
        "target_sha": "def",
        "repository": "org/repo",
        "draft": True,
    }
    context = main.build_queued_pr_context(metadata)
    assert context["draft"] is True
    assert context["files"] == []
    assert context["review_budget"]["total_files_seen"] == 0


def test_compare_findings_reports_added_removed_unchanged():
    old = [
        {"file": "a.py", "line": 1, "description": "same"},
        {"file": "b.py", "line": 2, "description": "old"},
    ]
    new = [
        {"file": "a.py", "line": 1, "description": "same"},
        {"file": "c.py", "line": 3, "description": "new"},
    ]
    comparison = main.compare_findings(old, new)
    assert comparison["counts"] == {"old": 2, "new": 2, "added": 1, "removed": 1, "unchanged": 1}
    assert comparison["added"][0]["file"] == "c.py"
    assert comparison["removed"][0]["file"] == "b.py"


def test_review_timeline_and_feedback_summary():
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    row = {
        "created_at": now,
        "updated_at": now,
        "repo": "org/repo",
        "pr_number": 7,
        "status": main.ReviewStatus.AWAITING_APPROVAL.value,
        "pr_context": {
            "action": "opened",
            "author": "alice",
        },
    }
    result = {
        "ranked_findings": [{"file": "app.py"}],
        "finding_feedback": {
            "0": {"verdict": "valid"},
            "1": {"verdict": "invalid"},
        },
    }
    timeline = main.review_timeline(row, result)
    assert [item["label"] for item in timeline] == [
        "Webhook received",
        "Analysis complete",
        "Awaiting reviewer action",
    ]
    assert main.feedback_summary(result) == {"valid": 1, "invalid": 1, "total": 2}


def test_rank_findings_uses_human_feedback_memory():
    base_issue = {
        "description": "Risky query",
        "file": "app.py",
        "line": 4,
        "severity": "medium",
        "confidence": 0.5,
        "finding_type": "possible_concern",
    }
    state = {
        "pr_context": {
            "human_feedback_memory": {
                "frequently_valid_categories": ["security"],
                "frequently_invalid_categories": ["performance"],
            }
        },
        "logic_issues": [],
        "security_issues": [{**base_issue, "description": "Security concern"}],
        "performance_issues": [{**base_issue, "description": "Performance concern"}],
        "deterministic_issues": [],
        "contract_issues": [],
        "test_evaluation": [],
    }
    ranked = agent.rank_findings(state)["ranked_findings"]
    assert ranked[0]["category"] == "security"
    assert ranked[0]["ranking_score"] > ranked[1]["ranking_score"]
    assert ranked[0]["effective_severity"] == "high"
    assert ranked[1]["effective_severity"] == "low"


def test_deterministic_checks_find_secret_and_repo_policy():
    state = {
        "pr_context": {
            "review_rules": {
                "policies": {
                    "blocked_paths": [{"path": "migrations/", "severity": "high"}],
                    "prohibited_patterns": [{"id": "repo.no_print", "pattern": r"print\(", "message": "No print", "severity": "low"}],
                }
            },
            "files": [{
                "filename": "migrations/001.sql",
                "added_line_details": [
                    {"line": 1, "content": "API_KEY = '123456789abc'"},
                    {"line": 2, "content": "print('debug')"},
                ],
                "removed_line_details": [],
            }],
        }
    }
    findings = agent.deterministic_checks(state)["deterministic_issues"]
    rule_ids = {finding["rule_id"] for finding in findings}
    assert {"repo.blocked_path", "deterministic.secret", "repo.no_print"} <= rule_ids


def test_diff_position_for_finding():
    pr_context = {
        "files": [{
            "filename": "app.py",
            "added_line_details": [{"line": 12, "content": "x = 1", "diff_position": 7}],
            "removed_line_details": [],
        }]
    }
    assert main.diff_position_for_finding(pr_context, {"file": "app.py", "line": 12}) == 7
