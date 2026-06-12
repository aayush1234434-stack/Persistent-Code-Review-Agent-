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
PR Diff
  -> Parser
  -> Context Builder
  -> Issue Detector
  -> Severity Scorer
  -> Review Generator
```

---

## Features

- Detects common issues such as security problems, bad practices, and policy violations
- Severity classification: HIGH / MEDIUM / LOW
- Benchmark test cases included
- Docker support
- GitHub Actions CI (pytest + benchmarks)
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

---

## Installation

```bash
git clone https://github.com/aayush1234434-stack/Persistent-Code-Review-Agent-.git
cd Persistent-Code-Review-Agent-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

---

## Run with Docker

```bash
docker compose up --build
```

---

## Project Structure

```text
agent.py              # Core agent logic
main.py               # FastAPI app and GitHub webhook handling
dashboard.html        # Review dashboard
benchmarks/           # Benchmark cases
tests/                # Unit and API integration tests
migrations/           # Database schema
.github/workflows/    # CI pipeline
```

---

## Future Improvements

- Larger benchmark suite with real PR examples
- More advanced LLM-based analysis
- Multi-agent review system
- Dedicated database table for human feedback
- Token and cost accounting

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
