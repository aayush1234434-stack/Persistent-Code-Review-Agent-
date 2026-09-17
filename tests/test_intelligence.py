import json
from pathlib import Path

import pytest
import yaml

import agent
from review_intelligence import (
    IntelligenceConfig,
    build_analysis_commands,
    build_symbol_dependency_map,
    parse_static_findings,
)


def test_production_requires_filesystem_sandbox():
    config = IntelligenceConfig(
        enabled=True,
        mode="local",
        sandbox_root=Path("/tmp/review-tests"),
        request_timeout_seconds=10,
        command_timeout_seconds=10,
        max_impacted_files=10,
        retain_workspace=False,
        allow_local_test_execution=False,
    )

    with pytest.raises(RuntimeError, match="isolated filesystem sandbox"):
        config.validate(production=True, worker_enabled=True)


def test_compose_sandbox_supervisor_has_only_required_workspace_capabilities():
    compose_path = Path(__file__).resolve().parents[1] / "docker-compose.yml"
    sandbox = yaml.safe_load(compose_path.read_text())["services"]["sandbox"]

    assert sandbox["network_mode"] == "none"
    assert sandbox["read_only"] is True
    assert sandbox["cap_drop"] == ["ALL"]
    assert set(sandbox["cap_add"]) == {
        "CHOWN",
        "DAC_OVERRIDE",
        "FOWNER",
        "KILL",
        "SETGID",
        "SETUID",
    }
    assert "no-new-privileges:true" in sandbox["security_opt"]


def test_symbol_map_limits_review_to_changed_files_and_callers(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "service.py").write_text(
        "def calculate(value):\n    return value * 2\n",
        encoding="utf-8",
    )
    (tmp_path / "app" / "consumer.py").write_text(
        "from app.service import calculate\n\ndef render():\n    return calculate(3)\n",
        encoding="utf-8",
    )
    (tmp_path / "app" / "unrelated.py").write_text("def untouched():\n    return 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_service.py").write_text(
        "from app.service import calculate\n\ndef test_calculate():\n    assert calculate(2) == 4\n",
        encoding="utf-8",
    )

    symbol_map = build_symbol_dependency_map(
        tmp_path,
        ["app/service.py"],
        max_impacted_files=10,
    )

    assert symbol_map["changed_files"] == ["app/service.py"]
    assert "app/consumer.py" in symbol_map["reverse_dependencies"]
    assert "app/consumer.py" in symbol_map["impacted_files"]
    assert "app/unrelated.py" not in symbol_map["impacted_files"]
    assert any(symbol["name"] == "calculate" for symbol in symbol_map["changed_symbols"])

    commands = build_analysis_commands(
        tmp_path,
        symbol_map,
        timeout_seconds=30,
        include_tests=True,
    )
    command_by_name = {command["name"]: command for command in commands}
    assert "app/unrelated.py" not in command_by_name["ruff"]["argv"]
    assert "tests/test_service.py" in command_by_name["tests"]["argv"]


def test_multilanguage_symbol_and_caller_maps(tmp_path):
    fixtures = {
        "typescript": {
            "changed": "ts/service.ts",
            "caller": "ts/consumer.ts",
            "changed_text": "export function calculate(value: number) { return value * 2; }\n",
            "caller_text": "import { calculate } from './service';\nexport const result = calculate(2);\n",
            "symbol": "calculate",
        },
        "go": {
            "changed": "go/pkg/service.go",
            "caller": "go/cmd/main.go",
            "changed_text": "package pkg\nfunc Calculate(value int) int { return value * 2 }\n",
            "caller_text": "package main\nimport \"example/go/pkg\"\nfunc main() { pkg.Calculate(2) }\n",
            "symbol": "Calculate",
        },
        "java": {
            "changed": "java/com/example/Service.java",
            "caller": "java/com/example/Consumer.java",
            "changed_text": "package com.example;\npublic class Service {\n public int calculate(int value) { return value * 2; }\n}\n",
            "caller_text": "package com.example;\nimport com.example.Service;\nclass Consumer { int run() { return new Service().calculate(2); } }\n",
            "symbol": "calculate",
        },
        "rust": {
            "changed": "rust/src/service.rs",
            "caller": "rust/src/main.rs",
            "changed_text": "pub fn calculate(value: i32) -> i32 { value * 2 }\n",
            "caller_text": "use crate::service::calculate;\nfn main() { calculate(2); }\n",
            "symbol": "calculate",
        },
    }
    for language, fixture in fixtures.items():
        changed_path = tmp_path / fixture["changed"]
        caller_path = tmp_path / fixture["caller"]
        changed_path.parent.mkdir(parents=True, exist_ok=True)
        caller_path.parent.mkdir(parents=True, exist_ok=True)
        changed_path.write_text(fixture["changed_text"])
        caller_path.write_text(fixture["caller_text"])

        symbol_map = build_symbol_dependency_map(
            tmp_path,
            [fixture["changed"]],
            max_impacted_files=20,
        )

        assert symbol_map["files"][fixture["changed"]]["language"] == language
        assert any(
            symbol["name"] == fixture["symbol"]
            for symbol in symbol_map["changed_symbols"]
        )
        assert fixture["caller"] in symbol_map["impacted_files"]

def test_static_tool_outputs_are_normalized():
    results = [
        {
            "name": "ruff",
            "stdout": json.dumps([
                {
                    "filename": "/sandbox/workspaces/one/repo/app.py",
                    "location": {"row": 7},
                    "message": "Undefined name",
                    "code": "F821",
                }
            ]),
        },
        {
            "name": "bandit",
            "stdout": json.dumps({
                "results": [{
                    "filename": "./security.py",
                    "line_number": 4,
                    "issue_text": "Use of eval",
                    "issue_severity": "HIGH",
                    "issue_confidence": "HIGH",
                    "test_id": "B307",
                    "code": "eval(value)",
                }]
            }),
        },
        {
            "name": "semgrep",
            "stdout": json.dumps({
                "results": [{
                    "path": "api.py",
                    "start": {"line": 9},
                    "check_id": "python-dangerous-eval",
                    "extra": {"message": "Dangerous eval", "severity": "ERROR"},
                }]
            }),
        },
    ]

    findings = parse_static_findings(results)

    assert [finding["file"] for finding in findings] == ["app.py", "security.py", "api.py"]
    assert findings[1]["severity"] == "high"
    assert findings[2]["evidence_source"] == "semgrep"


def test_evidence_corroboration_and_test_generation():
    pr_context = {
        "files": [{
            "filename": "app.py",
            "added_line_details": [{"line": 7, "content": "return eval(value)", "diff_position": 3}],
            "removed_line_details": [],
        }],
        "intelligence": {
            "static_findings": [{
                "file": "app.py",
                "line": 7,
                "description": "Dynamic evaluation",
                "evidence_source": "semgrep",
            }],
            "symbol_map": {
                "changed_symbols": [{
                    "file": "app.py",
                    "name": "parse_value",
                    "kind": "function",
                    "line": 5,
                    "end_line": 8,
                }]
            },
            "execution": {
                "tests": {"exit_code": 1, "output_excerpt": "FAILED tests/test_app.py app.py"}
            },
        },
    }
    finding = {
        "file": "app.py",
        "line": 7,
        "description": "Dynamic evaluation can execute input",
        "severity": "high",
        "category": "security",
        "confidence": 0.9,
    }

    verified = agent.verify_findings_grounded({
        "pr_context": pr_context,
        "ranked_findings": [finding],
    })

    assert verified["ranked_findings"][0]["verification_level"] == "corroborated"
    assert verified["ranked_findings"][0]["evidence_sources"] == [
        "diff",
        "semgrep",
        "execution:tests",
    ]
    suggestions = agent.generate_test_suggestions({
        "pr_context": pr_context,
        "ranked_findings": verified["ranked_findings"],
    })["test_suggestions"]
    assert suggestions[0]["target"] == "parse_value"
    assert "regression test" in suggestions[0]["title"].lower()
