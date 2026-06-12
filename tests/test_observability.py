from types import SimpleNamespace

import agent
import main


def test_merge_llm_usage_into_budget_tracks_tokens_and_cost():
    usage = agent.summarize_llm_usage([
        {
            "node": "logic_issues",
            "model": "gpt-4o-mini",
            "status": "success",
            "latency_ms": 120.0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "estimated_cost_usd": 0.000045,
        }
    ])
    budget = agent.merge_llm_usage_into_budget({"truncated": False}, usage)
    assert budget["llm_total_tokens"] == 150
    assert budget["llm_prompt_tokens"] == 100
    assert budget["llm_completion_tokens"] == 50
    assert budget["llm_estimated_cost_usd"] == 0.000045
    assert budget["llm_call_count"] == 1
    assert "llm_usage" in budget


def test_finalize_graph_result_merges_llm_usage():
    agent.reset_llm_usage()
    agent._llm_usage_tracker().append({
        "node": "extract_summary",
        "model": "gpt-4o-mini",
        "status": "success",
        "latency_ms": 80.0,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
        "estimated_cost_usd": 0.00001,
    })
    compact = main.finalize_graph_result({
        "pr_context": {"review_budget": {"truncated": False}},
        "merge_decision": {"decision": "approve", "reason": "ok"},
        "ranked_findings": [],
    })
    assert compact["review_budget"]["llm_total_tokens"] == 30


def test_extract_token_usage_from_response_metadata():
    response = SimpleNamespace(
        usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        response_metadata={},
    )
    usage = agent.extract_token_usage(response)
    assert usage == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    }


def test_merge_decision_to_check_conclusion():
    assert main.merge_decision_to_check_conclusion("approve") == "success"
    assert main.merge_decision_to_check_conclusion("reject") == "failure"
    assert main.merge_decision_to_check_conclusion("needs_review") == "neutral"


def test_dashboard_auth_required_in_production(monkeypatch):
    monkeypatch.setattr(main, "ENVIRONMENT", "production")
    assert main.dashboard_auth_required() is True


def test_format_review_comment_includes_llm_usage():
    comment = main.format_review_comment({
        "merge_decision": {"decision": "approve", "reason": "ok"},
        "ranked_findings": [],
        "review_budget": {
            "llm_total_tokens": 42,
            "llm_prompt_tokens": 30,
            "llm_completion_tokens": 12,
            "llm_estimated_cost_usd": 0.0012,
        },
    })
    assert "LLM usage" in comment
    assert "42 tokens" in comment
