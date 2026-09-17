"""Run the review agent over a saved corpus and persist raw predictions.

This command intentionally requires an explicitly configured LLM. Its output is
then scored by benchmarks/evaluate.py and can be retained as a regression
baseline for a prompt/model combination.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent
import main
from benchmarks.evaluate import load_jsonl, validate_corpus


def review_case(case: dict) -> dict:
    source = case["source"]
    file_name = case["metadata"]["file"]
    raw_diff = (
        f"diff --git a/{file_name} b/{file_name}\n"
        f"--- a/{file_name}\n"
        f"+++ b/{file_name}\n"
        f"{case['diff']}\n"
    )
    metadata = {
        "pr_number": source["pr_number"],
        "title": case["metadata"]["title"],
        "description": "Real-world benchmark case with a held human review label.",
        "author": "benchmark",
        "action": "opened",
        "url": source["url"],
        "source_branch": "benchmark-head",
        "source_sha": f"benchmark-{case['id']}",
        "target_branch": "main",
        "target_sha": "benchmark-base",
        "repository": source["repository"],
    }
    state = {"pr_context": main.build_pr_context(metadata, raw_diff, {})}
    started = time.perf_counter()
    agent.reset_llm_usage()
    for node in (agent.extract_summary, agent.classify_files):
        state.update(node(state))
    for node in (
        agent.logic_issues,
        agent.security_issues,
        agent.performance_issues,
        agent.contract_issues,
        agent.deterministic_checks,
        agent.static_analysis_checks,
    ):
        state.update(node(state))
    state.update(agent.test_evaluation(state))
    state.update(agent.rank_findings(state))
    state.update(agent.verify_findings_grounded(state))
    usage = agent.summarize_llm_usage()
    return {
        "case_id": case["id"],
        "findings": state.get("ranked_findings", []),
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "estimated_cost_usd": usage["totals"]["estimated_cost_usd"],
        "llm_usage": usage,
        "analysis_errors": state.get("analysis_errors", []),
    }


def main_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(__file__).parent / "corpus" / "real_world_prs.v1.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not getattr(agent.llm, "available", True):
        raise SystemExit("A configured LLM backend is required for a live benchmark run")
    cases = load_jsonl(args.corpus)
    errors = validate_corpus(cases)
    if errors:
        raise SystemExit("Invalid corpus: " + "; ".join(errors))
    if args.limit:
        cases = cases[: args.limit]

    existing = load_jsonl(args.output) if args.resume and args.output.exists() else []
    completed = {str(item["case_id"]) for item in existing}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if existing else "w"
    with args.output.open(mode) as output:
        for index, case in enumerate(cases, start=1):
            if case["id"] in completed:
                continue
            run = review_case(case)
            output.write(json.dumps(run, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{index}/{len(cases)}] {case['id']} ({len(run['findings'])} findings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
