# Persistent Code Review Agent

# Persistent Code Review Agent

An automated code review agent that analyzes pull requests, detects issues, and generates structured review feedback with severity ranking.

---

## 🚀 Problem

- Time-consuming  
- Inconsistent across reviewers  
- Prone to missing critical issues (e.g., security flaws, bad practices) 

---

## 💡 Solution

This project implements a **persistent code review agent** that:

* Parses pull request diffs
* Analyzes code for issues
* Assigns severity levels
* Generates structured review comments

---

## 🧠 How It Works

Pipeline:

PR Diff
→ Parser
→ Context Builder
→ Issue Detector
→ Severity Scorer
→ Review Generator

---

## 🔍 Features

* ✅ Detects common issues (security, bad practices, policy violations)
* 📊 Severity classification (HIGH / MEDIUM / LOW)
* 🧪 Benchmark test cases included
* 🐳 Docker support
* 📈 Extensible architecture

---

## 📦 Example

### Input (PR diff)

```diff
password = "123456"
```

### Output

* 🔴 HIGH: Hardcoded secret detected
  → Storing credentials directly in code is insecure

---

## 🧪 Benchmarks

| Case                  | Issues Detected |
| --------------------- | --------------- |
| Hardcoded Secret      | ✅               |
| Repo Policy Violation | ✅               |
| Clean Refactor        | No issues       |

---

## 🛠️ Installation

```bash
git clone https://github.com/aayush1234434-stack/Persistent-Code-Review-Agent-.git
cd Persistent-Code-Review-Agent-
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python main.py --pr-file benchmarks/prs/hardcoded_secret.json
```

---

## 🐳 Run with Docker

```bash
docker-compose up --build
```

---

## 🧩 Project Structure

```text
agent.py              # Core agent logic  
main.py               # Entry point  
benchmarks/           # Test cases  
tests/                # Unit tests  
migrations/           # DB schema  
```

---

## 📈 Future Improvements

* Integration with GitHub PR API
* Real-time PR commenting
* Advanced LLM-based analysis
* Multi-agent review system

---

## 🎯 Why This Matters

This project demonstrates:

* System design for AI agents
* Code analysis pipelines
* Structured output generation
* Real-world problem solving

---

## 👤 Author

Aayush Singh


The agent listens to GitHub PR webhooks, analyzes changed diffs with a staged review pipeline, stores review state in Postgres, and lets reviewers approve, rerun, mark findings, and post final GitHub review comments.

## Features

- GitHub PR webhook listener for opened, synchronized, reopened, and ready-for-review events
- LLM-assisted review graph for logic, security, performance, contract, and test coverage checks
- Deterministic rule checks for secrets, dynamic execution, bare exceptions, SQL interpolation, and repo policies
- Prompt-injection defenses for untrusted PR titles, descriptions, comments, filenames, and code
- Structured model output with schema validation
- Grounded findings with confidence, evidence, changed-line context, and severity ranking
- Inline GitHub review comments using diff positions
- Human feedback loop to mark findings valid or invalid and improve future ranking
- Repo-specific policy file support via `.github/pr-reviewer.yml`
- Draft PR queueing until the PR is marked ready for review
- Dashboard for triage, filtering, review timeline, checkpoint history, rerun comparison, and repository memory
- Local benchmark harness for measuring deterministic true positives and false positives
- Cost controls per repo for PR size, prompt budget, and model selection

## Architecture

```text
GitHub PR Webhook
        |
        v
FastAPI Service
        |
        v
Diff Parser + Budgeting + Repo Policies
        |
        v
LangGraph Review Pipeline
  - Summary
  - File classification
  - Logic review
  - Security review
  - Performance review
  - Contract review
  - Test review
  - Deterministic checks
  - Deduplication
  - Grounding verifier
  - Merge decision
        |
        v
Postgres State + Dashboard + GitHub Comments
```

## Tech Stack

- Python 3.11+
- FastAPI
- Postgres
- LangGraph
- LangChain OpenAI
- GitHub Webhooks/API
- Plain HTML/CSS/JS dashboard
- Pytest

## Prerequisites

- Python 3.11+
- PostgreSQL 14+
- GitHub personal access token with repository access
- OpenAI API key
- A GitHub repository where you can configure webhooks

## Local Setup

```bash
cd "/Users/aayushsingh/Desktop/PR review"

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Fill in:

```bash
GITHUB_WEBHOOK_SECRET=replace_me
GITHUB_TOKEN=replace_me
DATABASE_URL=postgresql://user:password@localhost:5432/pr_review
OPENAI_API_KEY=replace_me
DASHBOARD_API_KEY=replace_me
HTTP_TIMEOUT_SECONDS=20
OPENAI_REVIEW_MODEL=gpt-4o-mini
OPENAI_TRIAGE_MODEL=gpt-4o-mini
OPENAI_STRONG_REVIEW_MODEL=gpt-4o
REVIEW_PROMPT_CHAR_BUDGET=12000
```

Do not commit `.env`. It is ignored by `.gitignore`.

## Database Setup

Create your Postgres database, then run:

```bash
export $(grep -v '^#' .env | xargs)
psql "$DATABASE_URL" -f migrations/001_create_pr_reviews.sql
```

## Run Locally

```bash
export $(grep -v '^#' .env | xargs)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- Dashboard: `http://localhost:8000/dashboard?key=<DASHBOARD_API_KEY>`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/healthz`
- Readiness: `http://localhost:8000/readyz`

## GitHub Webhook Setup

For local webhook testing, expose your server:

```bash
ngrok http 8000
```

In your GitHub repository, create a webhook:

- Payload URL: `https://<your-ngrok-or-host>/github/webhook`
- Content type: `application/json`
- Secret: same value as `GITHUB_WEBHOOK_SECRET`
- Events: Pull requests

The agent reacts to:

- `opened`
- `synchronize`
- `reopened`
- `ready_for_review`

Draft PRs are queued and analyzed only after they become ready for review.

## Repo-Specific Policies

Add this file to a repository you want to review:

```text
.github/pr-reviewer.yml
```

Example:

```yaml
cost_controls:
  max_files: 20
  max_lines_per_file: 200
  max_total_lines: 1500
  prompt_char_budget: 12000
  triage_model: gpt-4o-mini
  strong_model: gpt-4o
  strong_model_min_changed_lines: 400

risky_extensions:
  - .py
  - .js
  - .ts
  - .sql

policies:
  blocked_paths:
    - path: migrations/
      severity: high
      message: Migration changes need database owner review.

  prohibited_patterns:
    - id: repo.no_print
      pattern: "print\\("
      severity: low
      message: Debug prints should not be committed.
```

Policy files are treated as untrusted input. They guide review behavior but are not allowed to override system instructions.

## Dashboard

The dashboard is designed for triage instead of raw JSON inspection.

It supports:

- Filtering by repo, PR number, status, author, decision, and search text
- Review status timeline
- Findings table with severity, confidence, evidence, category, and location
- Marking findings valid or invalid
- Posting final review comments
- Rerunning analysis from checkpoints
- Comparing old and rerun findings
- Viewing repository memory: risky files, authors, categories, and decisions

## Benchmarks

Run the deterministic benchmark suite:

```bash
python benchmarks/pr_review_benchmark.py
```

Benchmark fixtures live in:

```text
benchmarks/prs/
```

Current cases cover:

- Clean refactor with no findings
- Hardcoded secret detection
- Repo policy violation detection

Use this directory to grow a test set of known false positives and missed bugs over time.

## Tests

```bash
pytest -q
```

Current test coverage includes:

- GitHub signature validation
- Diff parsing and changed-line numbers
- Diff pruning and budget metadata
- Repo policy parsing
- Structured output validation
- Finding deduplication
- Grounding verifier
- Draft PR queue context
- Rerun comparison
- Timeline and feedback summaries
- Human feedback ranking adjustments
- Deterministic checks
- GitHub diff position lookup

## Docker Compose

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8000/dashboard?key=<DASHBOARD_API_KEY>
```

## Pushing to GitHub

From this directory:

```bash
git init
git add .
git commit -m "Initial persistent code review agent"
git branch -M main
git remote add origin https://github.com/aayush1234434-stack/Persistent-Code-Review-Agent-.git
git push -u origin main
```

If the remote already exists:

```bash
git remote set-url origin https://github.com/aayush1234434-stack/Persistent-Code-Review-Agent-.git
git push -u origin main
```

## Roadmap

- Add a larger benchmark suite with real PR examples
- Add integration tests for GitHub webhook to dashboard to comment posting
- Store human feedback in a dedicated table
- Add token and cost accounting from actual model responses
- Improve inline comment behavior for renamed files, deleted lines, and outdated commits
- Add CI for tests, benchmarks, linting, and basic security checks

## Security Notes

- Never commit `.env`
- Use a strong webhook secret
- Scope GitHub tokens as narrowly as possible
- Keep dashboard access behind `DASHBOARD_API_KEY`
- Treat PR code, filenames, comments, and policy files as untrusted input

## Status

This is a strong prototype and early production candidate. It is ready to push to GitHub and improve incrementally through issues and pull requests.
