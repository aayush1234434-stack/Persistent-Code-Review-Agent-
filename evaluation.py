"""Evaluation metrics, calibration, and prompt/model provenance.

The evaluator deliberately has no database or network dependency. Production
telemetry, checked-in benchmark snapshots, and ad-hoc model runs all use the
same metric implementation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

import agent

EVALUATION_SCHEMA_VERSION = "1.0"
CALIBRATION_BIN_COUNT = 10
PROMPT_FUNCTIONS = (
    agent.system_guard,
    agent.extract_summary,
    agent.classify_files,
    agent.finding_review_prompt,
    agent.logic_issues,
    agent.security_issues,
    agent.performance_issues,
    agent.contract_issues,
    agent.test_evaluation,
    agent.rank_findings,
    agent.verify_findings_grounded,
    agent.merge_decision,
)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def prompt_fingerprint() -> str:
    """Hash the actual prompt-building source and structured schemas.

    This turns an otherwise easy-to-forget prompt edit into a CI-visible
    contract change.
    """
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "functions": {function.__name__: inspect.getsource(function) for function in PROMPT_FUNCTIONS},
        "finding_schema": agent.schema_description(agent.FindingResult),
        "classification_schema": agent.schema_description(agent.FileClassificationResult),
    }
    return _stable_hash(payload)


def active_model_policy(policy: dict[str, Any] | None = None) -> dict[str, str]:
    policy = policy or {}
    review_model = str(
        policy.get("review_model")
        or policy.get("selected_model")
        or os.environ.get("OPENAI_REVIEW_MODEL", agent.DEFAULT_REVIEW_MODEL)
    )
    return {
        "triage_model": str(
            policy.get("triage_model")
            or os.environ.get("OPENAI_TRIAGE_MODEL", agent.DEFAULT_TRIAGE_MODEL)
        ),
        "review_model": review_model,
        "strong_model": str(
            policy.get("strong_model")
            or os.environ.get(
                "OPENAI_STRONG_REVIEW_MODEL",
                os.environ.get("OPENAI_REVIEW_MODEL", agent.DEFAULT_STRONG_REVIEW_MODEL),
            )
        ),
    }


def model_policy_fingerprint(policy: dict[str, Any] | None = None) -> str:
    return _stable_hash(active_model_policy(policy))


def evaluation_provenance(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "pricing_version": agent.MODEL_PRICING_VERSION,
        "prompt_fingerprint": prompt_fingerprint(),
        "model_policy_fingerprint": model_policy_fingerprint(policy),
        "model_policy": active_model_policy(policy),
    }


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(ordered[0], 6)
    rank = (len(ordered) - 1) * percentile_value
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(ordered[lower], 6)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
    return round(value, 6)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    return {
        "count": len(materialized),
        "mean": round(mean(materialized), 6) if materialized else None,
        "p50": percentile(materialized, 0.50),
        "p95": percentile(materialized, 0.95),
        "max": round(max(materialized), 6) if materialized else None,
    }


def _normalized_category(value: Any) -> str:
    category = str(value or "general").strip().lower()
    return category or "general"


def _category_set(value: Any) -> set[str]:
    normalized = _normalized_category(value)
    return {part.strip() for part in re.split(r"[/,+|]", normalized) if part.strip()}


def _finding_match_score(label: dict[str, Any], prediction: dict[str, Any]) -> int:
    label_id = str(label.get("id", ""))
    explicit = prediction.get("matched_label_ids") or []
    if label_id and label_id in {str(item) for item in explicit}:
        return 100

    if _category_set(label.get("category")).isdisjoint(_category_set(prediction.get("category"))):
        return -1
    label_file = str(label.get("file", ""))
    predicted_file = str(prediction.get("file", ""))
    if label_file and predicted_file and label_file != predicted_file:
        return -1
    label_line = label.get("line")
    prediction_line = prediction.get("line")
    if isinstance(label_line, int) and isinstance(prediction_line, int):
        distance = abs(label_line - prediction_line)
        if distance > 5:
            return -1
        return 50 - distance
    return 10


def match_case_findings(
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily pair findings using explicit label IDs or file/category/line."""
    candidates: list[tuple[int, int, int]] = []
    for label_index, label in enumerate(labels):
        for prediction_index, prediction in enumerate(predictions):
            score = _finding_match_score(label, prediction)
            if score >= 0:
                candidates.append((score, label_index, prediction_index))
    candidates.sort(reverse=True)
    matched_labels: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _score, label_index, prediction_index in candidates:
        if label_index in matched_labels or prediction_index in matched_predictions:
            continue
        matched_labels.add(label_index)
        matched_predictions.add(prediction_index)
        matches.append((label_index, prediction_index))
    unmatched_labels = [index for index in range(len(labels)) if index not in matched_labels]
    unmatched_predictions = [index for index in range(len(predictions)) if index not in matched_predictions]
    return matches, unmatched_labels, unmatched_predictions


def _metric_bucket(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_positive_rate": round(fp / (tp + fp), 6) if tp + fp else 0.0,
    }


def calibration_curve(samples: list[tuple[float, int]], bins: int = CALIBRATION_BIN_COUNT) -> dict[str, Any]:
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for confidence, outcome in samples:
        bounded = min(max(float(confidence), 0.0), 1.0)
        index = min(int(bounded * bins), bins - 1)
        buckets[index].append((bounded, int(bool(outcome))))

    points = []
    expected_calibration_error = 0.0
    total = len(samples)
    for index, bucket in enumerate(buckets):
        if bucket:
            average_confidence = mean(item[0] for item in bucket)
            empirical_accuracy = mean(item[1] for item in bucket)
            gap = abs(average_confidence - empirical_accuracy)
            expected_calibration_error += gap * len(bucket) / total
        else:
            average_confidence = None
            empirical_accuracy = None
            gap = None
        points.append({
            "lower": round(index / bins, 2),
            "upper": round((index + 1) / bins, 2),
            "count": len(bucket),
            "average_confidence": round(average_confidence, 6) if average_confidence is not None else None,
            "empirical_accuracy": round(empirical_accuracy, 6) if empirical_accuracy is not None else None,
            "gap": round(gap, 6) if gap is not None else None,
        })
    brier_score = mean((confidence - outcome) ** 2 for confidence, outcome in samples) if samples else None
    return {
        "sample_count": total,
        "expected_calibration_error": round(expected_calibration_error, 6) if samples else None,
        "brier_score": round(brier_score, 6) if brier_score is not None else None,
        "points": points,
    }


def evaluate_cases(cases: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    run_by_case = {str(run["case_id"]): run for run in runs}
    category_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    global_counts = {"tp": 0, "fp": 0, "fn": 0}
    confidence_samples: list[tuple[float, int]] = []
    durations: list[float] = []
    costs: list[float] = []
    clean_cases = 0
    noisy_clean_cases = 0
    feedback_valid = 0
    feedback_invalid = 0
    evaluated = 0
    valid_unlabeled = 0
    confirmed_false_positives = 0
    uncertain_unmatched = 0
    unadjudicated_unmatched = 0
    adjudicated_confidence_samples: list[tuple[float, int]] = []

    for case in cases:
        case_id = str(case["id"])
        run = run_by_case.get(case_id)
        if run is None:
            continue
        evaluated += 1
        labels = [dict(item) for item in case.get("labels", [])]
        findings = [dict(item) for item in run.get("findings", [])]
        matches, unmatched_labels, unmatched_predictions = match_case_findings(labels, findings)
        matched_prediction_indexes = {prediction_index for _label_index, prediction_index in matches}
        recovered_matches: list[tuple[int, int]] = []
        remaining_label_indexes = set(unmatched_labels)
        remaining_prediction_indexes = set(unmatched_predictions)
        adjudications: dict[int, dict[str, Any]] = {}
        for prediction_index in unmatched_predictions:
            adjudication = findings[prediction_index].get("benchmark_adjudication") or {}
            adjudications[prediction_index] = adjudication
            if adjudication.get("verdict") != "matches_label":
                continue
            matched_label_id = str(adjudication.get("matched_label_id") or "")
            label_index = next(
                (
                    candidate
                    for candidate in remaining_label_indexes
                    if str(labels[candidate].get("id") or "") == matched_label_id
                ),
                None,
            )
            if label_index is not None:
                recovered_matches.append((label_index, prediction_index))
                remaining_label_indexes.remove(label_index)
                remaining_prediction_indexes.remove(prediction_index)
                matched_prediction_indexes.add(prediction_index)

        for label_index, prediction_index in matches:
            category = _normalized_category(labels[label_index].get("category"))
            category_counts[category]["tp"] += 1
            global_counts["tp"] += 1
            confidence_samples.append((float(findings[prediction_index].get("confidence", 0.5)), 1))
            adjudicated_confidence_samples.append((float(findings[prediction_index].get("confidence", 0.5)), 1))
        for label_index, prediction_index in recovered_matches:
            category = _normalized_category(labels[label_index].get("category"))
            category_counts[category]["tp"] += 1
            global_counts["tp"] += 1
            confidence_samples.append((float(findings[prediction_index].get("confidence", 0.5)), 1))
            adjudicated_confidence_samples.append((float(findings[prediction_index].get("confidence", 0.5)), 1))
        for label_index in sorted(remaining_label_indexes):
            category = _normalized_category(labels[label_index].get("category"))
            category_counts[category]["fn"] += 1
            global_counts["fn"] += 1
        for prediction_index in sorted(remaining_prediction_indexes):
            category = _normalized_category(findings[prediction_index].get("category"))
            category_counts[category]["fp"] += 1
            global_counts["fp"] += 1
            confidence_samples.append((float(findings[prediction_index].get("confidence", 0.5)), 0))
            adjudication = adjudications.get(prediction_index, {})
            verdict = str(adjudication.get("verdict") or findings[prediction_index].get("benchmark_verdict") or "")
            confidence = float(findings[prediction_index].get("confidence", 0.5))
            if verdict == "valid_extra":
                valid_unlabeled += 1
                adjudicated_confidence_samples.append((confidence, 1))
            elif verdict == "false_positive":
                confirmed_false_positives += 1
                adjudicated_confidence_samples.append((confidence, 0))
            elif verdict == "uncertain":
                uncertain_unmatched += 1
            else:
                unadjudicated_unmatched += 1

        if not labels:
            clean_cases += 1
            if findings:
                noisy_clean_cases += 1
        if run.get("duration_ms") is not None:
            durations.append(float(run["duration_ms"]))
        if run.get("estimated_cost_usd") is not None:
            costs.append(float(run["estimated_cost_usd"]))
        for index, finding in enumerate(findings):
            verdict = str((finding.get("human_feedback") or {}).get("verdict", "")).lower()
            if verdict == "valid":
                feedback_valid += 1
            elif verdict in {"invalid", "dismissed"}:
                feedback_invalid += 1
            if index not in matched_prediction_indexes and verdict == "valid":
                # Human feedback remains useful even where the benchmark label set is incomplete.
                pass

    feedback_total = feedback_valid + feedback_invalid
    metrics = _metric_bucket(**global_counts)
    adjudicated_denominator = global_counts["tp"] + valid_unlabeled + confirmed_false_positives
    adjudicated_precision = (
        (global_counts["tp"] + valid_unlabeled) / adjudicated_denominator
        if adjudicated_denominator else None
    )
    unmatched_total = (
        valid_unlabeled + confirmed_false_positives + uncertain_unmatched + unadjudicated_unmatched
    )
    metrics.update({
        "case_count": len(cases),
        "evaluated_case_count": evaluated,
        "coverage": round(evaluated / len(cases), 6) if cases else 0.0,
        "clean_pr_false_positive_rate": round(noisy_clean_cases / clean_cases, 6) if clean_cases else None,
        "human_acceptance_rate": round(feedback_valid / feedback_total, 6) if feedback_total else None,
        "human_feedback_count": feedback_total,
        "time_to_review_ms": distribution(durations),
        "cost_per_pr_usd": distribution(costs),
        "calibration": calibration_curve(confidence_samples),
        "sparse_label_adjudication": {
            "valid_unlabeled": valid_unlabeled,
            "confirmed_false_positives": confirmed_false_positives,
            "uncertain": uncertain_unmatched,
            "unadjudicated": unadjudicated_unmatched,
            "coverage": round(
                (valid_unlabeled + confirmed_false_positives + uncertain_unmatched) / unmatched_total,
                6,
            ) if unmatched_total else 1.0,
            "precision": round(adjudicated_precision, 6) if adjudicated_precision is not None else None,
            "false_positive_rate": round(
                confirmed_false_positives / adjudicated_denominator,
                6,
            ) if adjudicated_denominator else None,
            "calibration": calibration_curve(adjudicated_confidence_samples),
        },
        "by_category": {
            category: _metric_bucket(**counts)
            for category, counts in sorted(category_counts.items())
        },
    })
    return metrics


def production_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    durations: list[float] = []
    costs: list[float] = []
    valid = 0
    invalid = 0
    category_feedback: dict[str, dict[str, int]] = defaultdict(lambda: {"valid": 0, "invalid": 0})
    for result in results:
        telemetry = result.get("evaluation_metrics", {}) or {}
        if telemetry.get("time_to_review_ms") is not None:
            durations.append(float(telemetry["time_to_review_ms"]))
        if telemetry.get("estimated_cost_usd") is not None:
            costs.append(float(telemetry["estimated_cost_usd"]))
        findings = result.get("ranked_findings", [])
        for index_text, feedback in (result.get("finding_feedback", {}) or {}).items():
            verdict = str(feedback.get("verdict", "")).lower()
            try:
                finding = findings[int(index_text)]
            except (IndexError, TypeError, ValueError):
                finding = {}
            category = _normalized_category(finding.get("category"))
            if verdict == "valid":
                valid += 1
                category_feedback[category]["valid"] += 1
            elif verdict in {"invalid", "dismissed"}:
                invalid += 1
                category_feedback[category]["invalid"] += 1
    feedback_total = valid + invalid
    return {
        "review_count": len(results),
        "time_to_review_ms": distribution(durations),
        "cost_per_pr_usd": distribution(costs),
        "human_acceptance_rate": round(valid / feedback_total, 6) if feedback_total else None,
        "human_feedback": {"valid": valid, "invalid": invalid, "total": feedback_total},
        "acceptance_by_category": {
            category: {
                **counts,
                "acceptance_rate": round(counts["valid"] / (counts["valid"] + counts["invalid"]), 6),
            }
            for category, counts in sorted(category_feedback.items())
        },
    }


def compare_to_baseline(report: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Return regression messages; an empty list means the gate passed."""
    failures: list[str] = []
    provenance = baseline.get("provenance", {})
    current = evaluation_provenance()
    for field in ("schema_version", "pricing_version", "prompt_fingerprint", "model_policy_fingerprint"):
        if provenance.get(field) != current.get(field):
            failures.append(f"{field} changed; run the 100+ case evaluation and accept a new baseline")

    thresholds = baseline.get("thresholds", {})
    checks = {
        "precision": report.get("precision"),
        "recall": report.get("recall"),
        "f1": report.get("f1"),
        "human_acceptance_rate": report.get("human_acceptance_rate"),
    }
    for name, actual in checks.items():
        minimum = thresholds.get(f"min_{name}")
        if minimum is not None and (actual is None or float(actual) < float(minimum)):
            failures.append(f"{name} {actual!r} is below minimum {minimum}")
    max_false_positive_rate = thresholds.get("max_false_positive_rate")
    if max_false_positive_rate is not None and float(report.get("false_positive_rate", 1.0)) > float(max_false_positive_rate):
        failures.append(
            f"false_positive_rate {report.get('false_positive_rate')} exceeds {max_false_positive_rate}"
        )
    max_ece = thresholds.get("max_expected_calibration_error")
    actual_ece = (report.get("calibration") or {}).get("expected_calibration_error")
    if max_ece is not None and (actual_ece is None or float(actual_ece) > float(max_ece)):
        failures.append(f"expected_calibration_error {actual_ece!r} exceeds {max_ece}")
    max_adjudicated_fpr = thresholds.get("max_adjudicated_false_positive_rate")
    actual_adjudicated_fpr = (report.get("sparse_label_adjudication") or {}).get("false_positive_rate")
    if max_adjudicated_fpr is not None and (
        actual_adjudicated_fpr is None or float(actual_adjudicated_fpr) > float(max_adjudicated_fpr)
    ):
        failures.append(
            f"adjudicated false_positive_rate {actual_adjudicated_fpr!r} exceeds {max_adjudicated_fpr}"
        )
    minimum_coverage = float(thresholds.get("min_coverage", 1.0))
    if float(report.get("coverage", 0.0)) < minimum_coverage:
        failures.append(f"coverage {report.get('coverage')} is below minimum {minimum_coverage}")
    return failures
