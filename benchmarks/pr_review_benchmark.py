import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent
import main


def run_case(path: Path) -> dict:
    case = json.loads(path.read_text())
    pr_context = main.build_pr_context(
        case["metadata"],
        case["diff"],
        case.get("review_rules", {}),
    )
    state = {"pr_context": pr_context}
    deterministic = agent.deterministic_checks(state)["deterministic_issues"]
    rule_ids = sorted({finding.get("rule_id") for finding in deterministic})
    expected = case.get("expected", {})
    expected_rule_ids = set(expected.get("rule_ids", []))
    missed_rules = sorted(expected_rule_ids - set(rule_ids))
    unexpected_rules = sorted(set(rule_ids) - expected_rule_ids)
    return {
        "name": case["name"],
        "findings": len(deterministic),
        "rule_ids": rule_ids,
        "missed_rules": missed_rules,
        "unexpected_rules": unexpected_rules,
        "passed": not missed_rules and len(deterministic) >= int(expected.get("min_findings", 0)),
    }


def main_cli() -> int:
    cases_dir = Path(__file__).parent / "prs"
    results = [run_case(path) for path in sorted(cases_dir.glob("*.json"))]
    print(json.dumps({"results": results}, indent=2))
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
