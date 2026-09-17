# Persistent Code Review Agent

An automated code review agent that analyzes pull requests, detects issues, and generates structured review feedback with severity ranking.

---

## Problem

Code reviews are:

- Time-consuming
- Inconsistent across reviewers
- Prone to missing critical issues, such as security flaws and bad practices

---

## Solution

This project implements a **persistent code review agent** that:

- Parses pull request diffs
- Analyzes code for issues
- Assigns severity levels
- Generates structured review comments

---

## How It Works

Pipeline:

```text
PR webhook -> durable queue -> isolated checkout
  -> symbol/dependency map -> impacted-file scope
  -> Ruff + Bandit + Semgrep + tests
  -> LLM + deterministic review -> evidence verifier
  -> human approval -> GitHub review
```

---

## Features

- Detects common issues such as security problems, bad practices, and policy violations
- Severity classification: HIGH / MEDIUM / LOW
- Benchmark test cases included
- Docker support
- GitHub Actions CI (pytest + benchmarks)
- Durable PostgreSQL job queue with worker leases and crash recovery
- Idempotent GitHub webhook ingestion
- Monotonic review versions for every PR commit
- Exponential retry/backoff with terminal failure state
- Transaction-safe human decisions and finding feedback
- Per-PR detached checkout in a network-isolated analysis container
- Python symbol, import, reverse-dependency, and caller mapping
- Cross-language repository intelligence for Python, JavaScript/TypeScript, Go, Java, and Rust
- Impacted-file linting and SAST with Ruff, Bandit, and offline Semgrep rules
- Targeted pytest execution with bounded resources and timeouts
- Finding corroboration from changed lines, static evidence, and execution output
- Structured regression-test suggestions for changed symbols and findings
- Inline GitHub review comments with safe suggested-patch blocks
- Developer-focused “why this matters” explanations
- One-click review of the latest PR commit from the dashboard
- Cross-version finding lifecycle: new, recurring, fixed, and dismissed
- Engineering-manager summaries with test status and priority risks
- File-level risk heatmaps and quantitative risk-baseline comparisons
- Versioned 120-case real-world PR evaluation corpus with auditable source URLs
- Precision, recall, F1, and false-positive rates globally and by issue category
- Review latency, estimated cost per PR, and human finding-acceptance telemetry
- Confidence calibration curves, Brier score, and expected calibration error
- Prompt/model fingerprints that force an explicit benchmark-baseline update in CI
- Extensible architecture

---

## Example

### Input: PR diff

```diff
+ password = "123456"
```

### Output

```text
HIGH: Hardcoded secret detected
Storing credentials directly in code is insecure.
```

---

## Benchmarks

| Case | Issues Detected |
| --- | --- |
| Hardcoded Secret | Yes |
| Repo Policy Violation | Yes |
| Clean Refactor | No issues |

Run benchmarks:

```bash
python benchmarks/pr_review_benchmark.py
```

The fast benchmark above exercises deterministic checks and pipeline behavior.
Phase 4 adds a separate trust evaluation over 120 unique, human-reviewed public
GitHub PRs. The corpus is pinned and checked in at
`benchmarks/corpus/real_world_prs.v1.jsonl`; every case includes its source PR
URL and upstream dataset revision.

Validate corpus provenance and the accepted prompt/model contract without an
LLM call:

```bash
python benchmarks/evaluate.py \
  --validate-only \
  --contract benchmarks/baselines/evaluation_contract.v1.json
```

Run the live agent over all cases, then calculate the quality, calibration,
latency, cost, and acceptance report:

```bash
python benchmarks/run_live_benchmark.py \
  --output benchmark-results/predictions.jsonl

python benchmarks/adjudicate_sparse_labels.py \
  --predictions benchmark-results/predictions.jsonl \
  --output benchmark-results/adjudicated-predictions.jsonl

python benchmarks/evaluate.py \
  --predictions benchmark-results/adjudicated-predictions.jsonl \
  --contract benchmarks/baselines/evaluation_contract.v1.json \
  --output-json benchmark-results/report.json \
  --output-markdown benchmark-results/report.md \
  --output-html docs/benchmark-report.html
```

The checked-in baseline is deliberately marked `contract_only` until the first
full live run is accepted; it does not pretend that bootstrap or oracle
predictions are model results. The manual **Live 120-PR Evaluation** workflow
runs the complete suite, independently adjudicates unmatched findings, publishes
the measured HTML report to GitHub Pages, and retains raw predictions and reports
for 90 days. Unmatched findings are reported separately as valid extras, confirmed
false positives, uncertain, or not yet adjudicated; sparse labels are not treated
as perfect ground truth.

---

## Installation

```bash
git clone https://github.com/aayush1234434-stack/Persistent-Code-Review-Agent-.git
cd Persistent-Code-Review-Agent-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-analysis.txt
```

Copy `.env.example` to `.env` and set your secrets before running the API.

---

## Usage

Run the test suite (LLM is mocked automatically; no `OPENAI_API_KEY` required):

```bash
pytest -q
```

Run benchmark cases (deterministic rules, ranking, grounding verifier, merge decision):

```bash
python benchmarks/pr_review_benchmark.py
```

CI runs both `pytest` and the benchmark script on every push/PR via GitHub Actions.

Run the FastAPI service:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The service applies SQL migrations during startup. By default it also runs one
embedded review worker. For separate web and worker processes, set
`REVIEW_WORKER_ENABLED=false` on the web process and run:

```bash
python worker.py
```

Webhook requests return only after the GitHub delivery is stored in
`review_jobs`. A worker claims jobs with a renewable lease; interrupted jobs
become claimable again after `REVIEW_JOB_LEASE_SECONDS`.

Local mode builds the repository map and runs static tools, but does not execute
repository tests unless `REVIEW_ALLOW_LOCAL_TEST_EXECUTION=true`. Production
requires `REVIEW_SANDBOX_MODE=filesystem`; the Docker Compose setup configures
this automatically.

---

## Run with Docker

```bash
docker compose up --build
```

The CI `docker-recovery` job also runs the destructive integration scenarios:

```bash
docker compose up --build -d db sandbox
python scripts/test_postgres_restart_recovery.py
docker compose run --rm worker python scripts/test_sandbox_roundtrip.py
docker compose down --volumes --remove-orphans
```

These verify PostgreSQL lease recovery after a database restart and a real
shared-volume sandbox round trip with loopback-only networking and resource
limits. They require a running Docker daemon.

---

## Project Structure

```text
agent.py              # Core agent logic
main.py               # FastAPI app and GitHub webhook handling
review_queue.py       # Durable queue, leases, retries, and migrations
review_intelligence.py # Checkout, repository map, tool parsing, evidence data
sandbox_service.py    # No-network, resource-limited analysis executor
semgrep-rules.yml     # Offline Semgrep policy (no runtime rule download)
worker.py             # Standalone worker entry point
dashboard.html        # Review dashboard
benchmarks/           # Benchmark cases
evaluation.py         # Quality, calibration, latency, cost, and acceptance metrics
tests/                # Unit and API integration tests
migrations/           # Database schema
.github/workflows/    # CI pipeline
docs/                 # GitHub Pages demo, report, and architecture diagram
scripts/              # Docker/PostgreSQL recovery integration checks
```

---

## Reliability Model

- GitHub delivery IDs are unique idempotency keys. If GitHub omits one, the
  signed payload hash is used.
- `(repo, PR number, source SHA)` uniquely identifies one review version.
- Workers use `FOR UPDATE SKIP LOCKED`, renewable leases, and bounded
  exponential backoff.
- A process can stop at any point; an unfinished job is recovered when its
  lease expires.
- Production startup fails when migrations or LangGraph PostgreSQL persistence
  are unavailable, or when an isolated analysis backend is not configured.

## Review Intelligence and Isolation

For each review version, the worker fetches only the PR head SHA into a unique
workspace. It maps changed symbols, direct importers, and callers, then limits
analysis to that impacted scope. The sandbox container has no network, runs
repository commands as a unique unprivileged per-review UID, receives no
GitHub/OpenAI secrets, and applies
CPU, memory, file-size, process-count, and wall-clock limits. Workspaces are
deleted after analysis unless `REVIEW_RETAIN_SANDBOX=true`.

Static findings are accepted only when they point to a changed file and survive
the changed-line grounding verifier. Test output and SAST results are attached
as corroborating evidence; they do not bypass diff grounding.

## Developer Experience

Each grounded finding includes its lifecycle, evidence, and a concise explanation
of operational impact. When the model or a static tool provides an exact bounded
replacement, the inline GitHub comment includes a native `suggestion` block that
developers can apply directly. Dismissed findings are persisted as reviewer
feedback, carried forward across review versions, excluded from merge decisions,
and omitted from inline publishing.

Every completed analysis also stores:

- A file-level weighted risk heatmap
- A comparison with the prior analyzed PR version
- Counts of new, recurring, fixed, and dismissed findings
- A manager summary covering scope, tests, priority risks, and risk direction

The dashboard’s **Review New Commits** action checks GitHub’s current PR head and
durably queues a new version only when the SHA changed. The same manager payload
is available from `GET /reviews/{id}/manager-summary`.

The public product preview is available from `docs/index.html`, with the
architecture diagram in `docs/architecture.svg`. The `pages.yml` workflow
publishes the demo on pushes to `main`; the manual live-evaluation workflow
replaces the report placeholder with measured results after credentials are
configured. Demo cards are explicitly labelled illustrative until that run.

## Evaluation and Trust

Every analyzed review stores `evaluation_metrics` alongside its result:

- End-to-end time to review
- Estimated LLM cost from token usage and configured model pricing
- Exact prompt fingerprint and active model-policy fingerprint
- The triage, review, and strong-model policy used for that review

`GET /evaluation/metrics` aggregates production p50/p95 latency, cost per PR,
overall human acceptance, and acceptance by category. Prometheus also exports
review-time, per-review cost, and finding-feedback series. The dashboard shows
the production summary; benchmark-only metrics such as recall and calibration
stay in the versioned evaluation report because production reviews have no
complete ground-truth label set.

CI hashes the actual prompt-building functions and structured schemas, plus the
default model policy. A prompt, schema, or model-policy change therefore fails
the evaluation-contract check until a maintainer runs the 120-case suite,
reviews the category-level regressions, and explicitly accepts a new contract.

## Future Improvements

- Run and publish the authenticated 120-case baseline, including human review of
  judge-adjudicated extras
- Add parser adapters for additional languages and dependency ecosystems
- Add a hosted review history with organization-level benchmark trends

---

## Why This Matters

This project demonstrates:

- System design for AI agents
- Code analysis pipelines
- Structured output generation
- Real-world problem solving

---

## Author

Aayush Singh
