"""Evaluate saved review runs against the versioned real-world PR corpus."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import compare_to_baseline, evaluate_cases, evaluation_provenance


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def corpus_fingerprint(cases: list[dict[str, Any]]) -> str:
    canonical = "\n".join(
        json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for case in cases
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_corpus(cases: list[dict[str, Any]], minimum_cases: int = 100) -> list[str]:
    errors: list[str] = []
    if len(cases) < minimum_cases:
        errors.append(f"corpus contains {len(cases)} cases; at least {minimum_cases} are required")
    ids = [str(case.get("id", "")) for case in cases]
    if len(set(ids)) != len(ids):
        errors.append("case IDs must be unique")
    pr_keys = []
    for index, case in enumerate(cases):
        label = f"case[{index}]"
        if not case.get("id"):
            errors.append(f"{label} has no id")
        source = case.get("source") or {}
        if source.get("type") != "real_github_pr_human_review":
            errors.append(f"{label} is not marked as a human-reviewed real GitHub PR")
        if not source.get("dataset") or not source.get("dataset_revision"):
            errors.append(f"{label} has no pinned dataset provenance")
        repository = str(source.get("repository", ""))
        pr_number = source.get("pr_number")
        if not repository or not isinstance(pr_number, int):
            errors.append(f"{label} has incomplete PR provenance")
        else:
            pr_keys.append((repository, pr_number))
            expected_url = f"https://github.com/{repository}/pull/{pr_number}"
            if source.get("url") != expected_url:
                errors.append(f"{label} has an invalid PR URL")
        if not case.get("diff"):
            errors.append(f"{label} has no changed diff")
        labels = case.get("labels")
        if not isinstance(labels, list) or not labels:
            errors.append(f"{label} has no human-reviewed labels")
        elif not all(isinstance(item, dict) and item.get("human_verified") is True for item in labels):
            errors.append(f"{label} contains a label that is not human verified")
    if len(set(pr_keys)) != len(pr_keys):
        errors.append("corpus must contain at most one case per repository/PR pair")
    return errors


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# PR review evaluation",
        "",
        f"Cases evaluated: **{report['evaluated_case_count']}/{report['case_count']}**",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Precision | {report['precision']:.3f} |",
        f"| Recall | {report['recall']:.3f} |",
        f"| F1 | {report['f1']:.3f} |",
        f"| Finding false-positive rate | {report['false_positive_rate']:.3f} |",
        f"| Clean-PR false-positive rate | {report['clean_pr_false_positive_rate'] if report['clean_pr_false_positive_rate'] is not None else 'n/a'} |",
        f"| Human acceptance rate | {report['human_acceptance_rate'] if report['human_acceptance_rate'] is not None else 'n/a'} |",
        f"| Mean review time (ms) | {report['time_to_review_ms']['mean'] if report['time_to_review_ms']['mean'] is not None else 'n/a'} |",
        f"| Mean cost / PR (USD) | {report['cost_per_pr_usd']['mean'] if report['cost_per_pr_usd']['mean'] is not None else 'n/a'} |",
        f"| Expected calibration error | {report['calibration']['expected_calibration_error'] if report['calibration']['expected_calibration_error'] is not None else 'n/a'} |",
        f"| Adjudicated precision | {report['sparse_label_adjudication']['precision'] if report['sparse_label_adjudication']['precision'] is not None else 'n/a'} |",
        f"| Confirmed false-positive rate | {report['sparse_label_adjudication']['false_positive_rate'] if report['sparse_label_adjudication']['false_positive_rate'] is not None else 'n/a'} |",
        f"| Sparse-label adjudication coverage | {report['sparse_label_adjudication']['coverage']:.3f} |",
        "",
        "## Category quality",
        "",
        "| Category | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, values in report["by_category"].items():
        lines.append(
            f"| {category} | {values['tp']} | {values['fp']} | {values['fn']} | "
            f"{values['precision']:.3f} | {values['recall']:.3f} | {values['f1']:.3f} |"
        )
    lines.extend([
        "",
        "## Confidence calibration",
        "",
        "| Confidence bin | Findings | Mean confidence | Empirical accuracy | Gap |",
        "|---|---:|---:|---:|---:|",
    ])
    for point in report["calibration"]["points"]:
        lines.append(
            f"| {point['lower']:.1f}–{point['upper']:.1f} | {point['count']} | "
            f"{point['average_confidence'] if point['average_confidence'] is not None else 'n/a'} | "
            f"{point['empirical_accuracy'] if point['empirical_accuracy'] is not None else 'n/a'} | "
            f"{point['gap'] if point['gap'] is not None else 'n/a'} |"
        )
    return "\n".join(lines) + "\n"


def html_report(report: dict[str, Any]) -> str:
    categories = "".join(
        "<tr>"
        f"<td>{html.escape(category)}</td><td>{values['tp']}</td><td>{values['fp']}</td><td>{values['fn']}</td>"
        f"<td>{values['precision']:.3f}</td><td>{values['recall']:.3f}</td><td>{values['f1']:.3f}</td>"
        "</tr>"
        for category, values in report["by_category"].items()
    )
    calibration = "".join(
        "<tr>"
        f"<td>{point['lower']:.1f}–{point['upper']:.1f}</td><td>{point['count']}</td>"
        f"<td>{point['average_confidence'] if point['average_confidence'] is not None else 'n/a'}</td>"
        f"<td>{point['empirical_accuracy'] if point['empirical_accuracy'] is not None else 'n/a'}</td>"
        f"<td>{point['gap'] if point['gap'] is not None else 'n/a'}</td>"
        "</tr>"
        for point in report["calibration"]["points"]
    )
    adjudicated = report["sparse_label_adjudication"]
    provenance = report["provenance"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PR Review Evaluation Report</title><style>
body{{max-width:1100px;margin:50px auto;padding:0 24px;background:#07111f;color:#eaf4ff;font:15px/1.5 system-ui}}a{{color:#38bdf8}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card{{background:#0e1d30;border:1px solid #29425d;border-radius:12px;padding:18px}}
.card b{{display:block;font-size:26px;color:#6ee7d5}}table{{width:100%;border-collapse:collapse;background:#0e1d30}}th,td{{padding:10px;border:1px solid #29425d;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{color:#8eddf7}}@media(max-width:760px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><a href="index.html">← Product demo</a><h1>Real-world PR evaluation</h1>
<p>Measured across {report['evaluated_case_count']} of {report['case_count']} pinned public pull-request cases.</p>
<div class="grid"><div class="card">Precision<b>{report['precision']:.3f}</b></div><div class="card">Recall<b>{report['recall']:.3f}</b></div><div class="card">F1<b>{report['f1']:.3f}</b></div><div class="card">Strict FPR<b>{report['false_positive_rate']:.3f}</b></div></div>
<h2>Sparse-label adjudication</h2><div class="grid"><div class="card">Adjudicated precision<b>{adjudicated['precision'] if adjudicated['precision'] is not None else 'n/a'}</b></div><div class="card">Confirmed FPR<b>{adjudicated['false_positive_rate'] if adjudicated['false_positive_rate'] is not None else 'n/a'}</b></div><div class="card">Valid extras<b>{adjudicated['valid_unlabeled']}</b></div><div class="card">Coverage<b>{adjudicated['coverage']:.3f}</b></div></div>
<h2>Quality by category</h2><table><thead><tr><th>Category</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{categories}</tbody></table>
<h2>Confidence calibration</h2><p>ECE: <b>{report['calibration']['expected_calibration_error']}</b> · Brier score: <b>{report['calibration']['brier_score']}</b></p><table><thead><tr><th>Bin</th><th>N</th><th>Confidence</th><th>Accuracy</th><th>Gap</th></tr></thead><tbody>{calibration}</tbody></table>
<h2>Efficiency and trust</h2><div class="grid"><div class="card">P50 time (ms)<b>{report['time_to_review_ms']['p50'] or 'n/a'}</b></div><div class="card">P95 time (ms)<b>{report['time_to_review_ms']['p95'] or 'n/a'}</b></div><div class="card">Mean cost / PR<b>{report['cost_per_pr_usd']['mean'] or 'n/a'}</b></div><div class="card">Human acceptance<b>{report['human_acceptance_rate'] if report['human_acceptance_rate'] is not None else 'n/a'}</b></div></div>
<h2>Reproducibility</h2><p><code>Corpus {html.escape(report['corpus_fingerprint'])}</code><br><code>Prompt {html.escape(provenance['prompt_fingerprint'])}</code><br><code>Model policy {html.escape(provenance['model_policy_fingerprint'])}</code><br><code>Pricing {html.escape(provenance['pricing_version'])}</code></p>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(__file__).parent / "corpus" / "real_world_prs.v1.jsonl")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    cases = load_jsonl(args.corpus)
    errors = validate_corpus(cases)
    if args.contract:
        contract = json.loads(args.contract.read_text())
        actual_corpus_fingerprint = corpus_fingerprint(cases)
        if contract.get("corpus_fingerprint") != actual_corpus_fingerprint:
            errors.append("corpus_fingerprint changed; review the corpus and accept a new evaluation contract")
        expected = contract.get("provenance", {})
        actual = evaluation_provenance()
        for field in ("schema_version", "pricing_version", "prompt_fingerprint", "model_policy_fingerprint"):
            if expected.get(field) != actual.get(field):
                errors.append(f"{field} changed; run the corpus and accept a new evaluation contract")
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    if args.validate_only:
        print(json.dumps({
            "ok": True,
            "cases": len(cases),
            "corpus_fingerprint": corpus_fingerprint(cases),
            "provenance": evaluation_provenance(),
        }, indent=2))
        return 0
    if not args.predictions:
        parser.error("--predictions is required unless --validate-only is used")

    runs = load_jsonl(args.predictions)
    report = evaluate_cases(cases, runs)
    report["corpus_fingerprint"] = corpus_fingerprint(cases)
    report["provenance"] = evaluation_provenance()
    failures = compare_to_baseline(report, json.loads(args.contract.read_text())) if args.contract else []
    report["regression_gate"] = {"passed": not failures, "failures": failures}
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown_report(report))
    if args.output_html:
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        args.output_html.write_text(html_report(report))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
