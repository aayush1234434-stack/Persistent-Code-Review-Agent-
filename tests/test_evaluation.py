import json
from pathlib import Path

import pytest

import evaluation
import main
from benchmarks.evaluate import corpus_fingerprint, load_jsonl, validate_corpus


def test_real_world_corpus_has_120_unique_auditable_prs():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "corpus" / "real_world_prs.v1.jsonl"
    cases = load_jsonl(path)

    assert len(cases) == 120
    assert validate_corpus(cases) == []
    assert len({(case["source"]["repository"], case["source"]["pr_number"]) for case in cases}) == 120
    assert {case["labels"][0]["category"] for case in cases} == {
        "logic", "tests", "performance", "security", "concurrency", "contract", "data"
    }


def test_evaluation_computes_quality_cost_latency_acceptance_and_calibration():
    cases = [
        {
            "id": "one",
            "labels": [
                {"id": "security-1", "category": "security", "file": "app.py", "line": 10},
                {"id": "performance-1", "category": "performance", "file": "app.py", "line": 20},
            ],
        },
        {"id": "two", "labels": [{"id": "logic-1", "category": "logic", "file": "api.py", "line": 5}]},
    ]
    runs = [
        {
            "case_id": "one",
            "duration_ms": 100,
            "estimated_cost_usd": 0.01,
            "findings": [
                {
                    "category": "security", "file": "app.py", "line": 10, "confidence": 0.9,
                    "human_feedback": {"verdict": "valid"},
                },
                {
                    "category": "logic", "file": "other.py", "line": 1, "confidence": 0.2,
                    "human_feedback": {"verdict": "invalid"},
                },
            ],
        },
        {"case_id": "two", "duration_ms": 300, "estimated_cost_usd": 0.03, "findings": []},
    ]

    report = evaluation.evaluate_cases(cases, runs)

    assert report["tp"] == 1
    assert report["fp"] == 1
    assert report["fn"] == 2
    assert report["precision"] == 0.5
    assert report["recall"] == pytest.approx(1 / 3, abs=1e-6)
    assert report["by_category"]["security"]["recall"] == 1.0
    assert report["by_category"]["performance"]["recall"] == 0.0
    assert report["false_positive_rate"] == 0.5
    assert report["human_acceptance_rate"] == 0.5
    assert report["time_to_review_ms"]["mean"] == 200.0
    assert report["time_to_review_ms"]["p95"] == 290.0
    assert report["cost_per_pr_usd"]["mean"] == 0.02
    assert report["calibration"]["expected_calibration_error"] == 0.15
    assert report["calibration"]["brier_score"] == 0.025


def test_composite_prediction_category_matches_specific_ground_truth_category():
    matches, missed, unexpected = evaluation.match_case_findings(
        [{"id": "one", "category": "security", "file": "app.py"}],
        [{"category": "logic/security", "file": "app.py", "confidence": 0.9}],
    )
    assert matches == [(0, 0)]
    assert missed == []
    assert unexpected == []


def test_sparse_label_adjudication_separates_valid_extras_from_false_positives():
    report = evaluation.evaluate_cases(
        [{"id": "one", "labels": [{"id": "label-1", "category": "logic", "file": "app.py"}]}],
        [{
            "case_id": "one",
            "findings": [
                {"category": "logic", "file": "app.py", "confidence": 0.9},
                {
                    "category": "security", "file": "app.py", "confidence": 0.8,
                    "benchmark_adjudication": {"verdict": "valid_extra"},
                },
                {
                    "category": "tests", "file": "app.py", "confidence": 0.7,
                    "benchmark_adjudication": {"verdict": "false_positive"},
                },
                {
                    "category": "contract", "file": "app.py", "confidence": 0.6,
                    "benchmark_adjudication": {"verdict": "uncertain"},
                },
            ],
        }],
    )

    assert report["false_positive_rate"] == 0.75
    adjudicated = report["sparse_label_adjudication"]
    assert adjudicated["valid_unlabeled"] == 1
    assert adjudicated["confirmed_false_positives"] == 1
    assert adjudicated["uncertain"] == 1
    assert adjudicated["unadjudicated"] == 0
    assert adjudicated["coverage"] == 1.0
    assert adjudicated["precision"] == pytest.approx(2 / 3, abs=1e-6)
    assert adjudicated["false_positive_rate"] == pytest.approx(1 / 3, abs=1e-6)


def test_sparse_label_adjudicator_can_recover_a_strict_unmatched_label():
    report = evaluation.evaluate_cases(
        [{"id": "one", "labels": [{"id": "label-1", "category": "security", "file": "app.py", "line": 40}]}],
        [{
            "case_id": "one",
            "findings": [{
                "category": "security", "file": "app.py", "line": 10, "confidence": 0.8,
                "benchmark_adjudication": {
                    "verdict": "matches_label", "matched_label_id": "label-1",
                },
            }],
        }],
    )

    assert report["tp"] == 1
    assert report["fp"] == 0
    assert report["fn"] == 0


def test_production_metrics_use_only_recorded_telemetry_and_feedback():
    metrics = evaluation.production_metrics([
        {
            "evaluation_metrics": {"time_to_review_ms": 200, "estimated_cost_usd": 0.02},
            "ranked_findings": [{"category": "security"}, {"category": "logic"}],
            "finding_feedback": {
                "0": {"verdict": "valid"},
                "1": {"verdict": "dismissed"},
            },
        },
        {"evaluation_metrics": {"time_to_review_ms": 400, "estimated_cost_usd": 0.04}},
    ])

    assert metrics["review_count"] == 2
    assert metrics["time_to_review_ms"]["p50"] == 300.0
    assert metrics["cost_per_pr_usd"]["mean"] == 0.03
    assert metrics["human_acceptance_rate"] == 0.5
    assert metrics["acceptance_by_category"]["security"]["acceptance_rate"] == 1.0
    assert metrics["acceptance_by_category"]["logic"]["acceptance_rate"] == 0.0


def test_review_result_records_end_to_end_time_cost_and_provenance(monkeypatch):
    monkeypatch.setattr(main.time, "perf_counter", lambda: 11.25)
    result = main.attach_evaluation_metrics(
        {"review_budget": {"llm_estimated_cost_usd": 0.1234567}},
        started_at=10.0,
        model_policy={"selected_model": "gpt-4.1-mini", "triage_model": "gpt-4.1-nano", "strong_model": "gpt-4.1"},
    )

    telemetry = result["evaluation_metrics"]
    assert telemetry["time_to_review_ms"] == 1250.0
    assert telemetry["estimated_cost_usd"] == 0.123457
    assert telemetry["model_policy"]["review_model"] == "gpt-4.1-mini"
    assert len(telemetry["prompt_fingerprint"]) == 64


def test_regression_gate_detects_contract_and_metric_regressions():
    baseline = {
        "provenance": evaluation.evaluation_provenance(),
        "thresholds": {
            "min_precision": 0.8,
            "min_recall": 0.7,
            "min_coverage": 1.0,
            "max_false_positive_rate": 0.2,
            "max_expected_calibration_error": 0.1,
        },
    }
    report = {
        "precision": 0.75,
        "recall": 0.8,
        "f1": 0.77,
        "coverage": 1.0,
        "false_positive_rate": 0.25,
        "human_acceptance_rate": None,
        "calibration": {"expected_calibration_error": 0.15},
    }
    failures = evaluation.compare_to_baseline(report, baseline)
    assert any("precision" in failure for failure in failures)
    assert any("false_positive_rate" in failure for failure in failures)
    assert any("expected_calibration_error" in failure for failure in failures)

    baseline["provenance"]["prompt_fingerprint"] = "changed"
    assert any("prompt_fingerprint changed" in failure for failure in evaluation.compare_to_baseline(report, baseline))


def test_checked_in_evaluation_contract_matches_prompt_and_model_policy():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "baselines" / "evaluation_contract.v1.json"
    baseline = json.loads(path.read_text())
    assert baseline["provenance"] == evaluation.evaluation_provenance()
    cases = load_jsonl(Path(__file__).resolve().parents[1] / baseline["corpus"])
    assert baseline["corpus_fingerprint"] == corpus_fingerprint(cases)
