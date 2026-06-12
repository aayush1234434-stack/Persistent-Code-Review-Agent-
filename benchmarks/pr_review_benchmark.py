import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent
import main


def run_deterministic_case(case: dict, pr_context: dict) -> dict:
    state = {"pr_context": pr_context}
    deterministic = agent.deterministic_checks(state)["deterministic_issues"]
    rule_ids = sorted({finding.get("rule_id") for finding in deterministic if finding.get("rule_id")})
    expected = case.get("expected", {})
    expected_rule_ids = set(expected.get("rule_ids", []))
    missed_rules = sorted(expected_rule_ids - set(rule_ids))
    unexpected_rules = sorted(set(rule_ids) - expected_rule_ids)
    if expected_rule_ids:
        passed = not missed_rules and not unexpected_rules
    else:
        passed = not unexpected_rules
    if "min_deterministic_findings" in expected:
        passed = passed and len(deterministic) >= int(expected["min_deterministic_findings"])
    return {
        "findings": len(deterministic),
        "rule_ids": rule_ids,
        "missed_rules": missed_rules,
        "unexpected_rules": unexpected_rules,
        "passed": passed,
    }


def run_analysis_pipeline(case: dict, pr_context: dict) -> dict:
    state = {"pr_context": pr_context}
    state.update(agent.deterministic_checks(state))

    injected = case.get("injected_issues", {})
    for key, value in injected.items():
        state[key] = value

    state.update(agent.rank_findings(state))
    state.update(agent.verify_findings_grounded(state))
    state.update(agent.merge_decision(state))

    ranked = state.get("ranked_findings", [])
    grounding = state.get("grounding_summary", {})
    decision = state.get("merge_decision", {}).get("decision")
    expected = case.get("expected", {})

    checks = {
        "merge_decision": decision == expected.get("merge_decision", decision),
        "min_verified": grounding.get("verified", 0) >= int(expected.get("min_verified", 0)),
        "max_dropped": grounding.get("dropped", 0) <= int(expected.get("max_dropped", 999)),
        "min_dropped": grounding.get("dropped", 0) >= int(expected.get("min_dropped", 0)),
        "min_ranked": len(ranked) >= int(expected.get("min_findings", 0)),
    }

    if expected.get("top_category"):
        checks["top_category"] = ranked[0].get("category") == expected["top_category"] if ranked else False

    return {
        "ranked_findings": len(ranked),
        "top_category": ranked[0].get("category") if ranked else None,
        "merge_decision": decision,
        "grounding_summary": grounding,
        "passed": all(checks.values()),
        "checks": checks,
    }


def run_case(path: Path) -> dict:
    case = json.loads(path.read_text())
    pr_context = main.build_pr_context(
        case["metadata"],
        case["diff"],
        case.get("review_rules", {}),
    )
    deterministic = run_deterministic_case(case, pr_context)
    pipeline = run_analysis_pipeline(case, pr_context)
    passed = deterministic["passed"] and pipeline["passed"]
    return {
        "name": case["name"],
        "deterministic": deterministic,
        "pipeline": pipeline,
        "passed": passed,
    }


def main_cli() -> int:
    cases_dir = Path(__file__).parent / "prs"
    results = [run_case(path) for path in sorted(cases_dir.glob("*.json"))]
    print(json.dumps({"results": results}, indent=2))
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
