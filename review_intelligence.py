"""Repository-aware review intelligence and isolated analysis orchestration."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_SOURCE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rs"}
MAX_COMMAND_OUTPUT = 200_000


@dataclass(frozen=True)
class IntelligenceConfig:
    enabled: bool
    mode: str
    sandbox_root: Path
    request_timeout_seconds: int
    command_timeout_seconds: int
    max_impacted_files: int
    retain_workspace: bool
    allow_local_test_execution: bool

    @classmethod
    def from_environment(cls) -> "IntelligenceConfig":
        truthy = {"1", "true", "yes"}
        return cls(
            enabled=os.environ.get("REVIEW_INTELLIGENCE_ENABLED", "true").strip().lower() in truthy,
            mode=os.environ.get("REVIEW_SANDBOX_MODE", "local").strip().lower(),
            sandbox_root=Path(
                os.environ.get(
                    "REVIEW_SANDBOX_ROOT",
                    str(Path(tempfile.gettempdir()) / "pr-review-sandbox"),
                )
            ),
            request_timeout_seconds=int(os.environ.get("REVIEW_SANDBOX_REQUEST_TIMEOUT_SECONDS", "900")),
            command_timeout_seconds=int(os.environ.get("REVIEW_SANDBOX_COMMAND_TIMEOUT_SECONDS", "300")),
            max_impacted_files=int(os.environ.get("REVIEW_MAX_IMPACTED_FILES", "40")),
            retain_workspace=os.environ.get("REVIEW_RETAIN_SANDBOX", "false").strip().lower() in truthy,
            allow_local_test_execution=os.environ.get(
                "REVIEW_ALLOW_LOCAL_TEST_EXECUTION", "false"
            ).strip().lower() in truthy,
        )

    def validate(self, *, production: bool, worker_enabled: bool) -> None:
        if not self.enabled or not worker_enabled:
            return
        if self.mode not in {"filesystem", "local"}:
            raise RuntimeError("REVIEW_SANDBOX_MODE must be filesystem or local")
        if production and self.mode != "filesystem":
            raise RuntimeError("Production review intelligence requires the isolated filesystem sandbox")
        if self.request_timeout_seconds < 1 or self.command_timeout_seconds < 1:
            raise RuntimeError("Sandbox timeouts must be positive")
        if self.max_impacted_files < 1:
            raise RuntimeError("REVIEW_MAX_IMPACTED_FILES must be positive")


def _minimal_subprocess_env(github_token: str | None = None) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": tempfile.gettempdir(),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_TERMINAL_PROMPT": "0",
    }
    if github_token:
        env.update(
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.extraHeader",
                "GIT_CONFIG_VALUE_0": f"Authorization: Bearer {github_token}",
            }
        )
    return env


def _run_checkout_step(argv: list[str], *, env: dict[str, str], timeout: int = 120) -> None:
    result = subprocess.run(
        argv,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "git command failed")[-1000:]
        raise RuntimeError(message)


def clone_pr_head(
    repository: str,
    source_sha: str,
    destination: Path,
    github_token: str | None,
) -> None:
    """Fetch only the requested PR commit without putting credentials in argv."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid GitHub repository name")
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", source_sha):
        raise ValueError("Invalid pull-request source SHA")

    destination.mkdir(parents=True, exist_ok=False)
    env = _minimal_subprocess_env(github_token)
    remote_url = f"https://github.com/{repository}.git"
    _run_checkout_step(["git", "init", "--quiet", str(destination)], env=env)
    _run_checkout_step(
        ["git", "-C", str(destination), "remote", "add", "origin", remote_url],
        env=env,
    )
    _run_checkout_step(
        ["git", "-C", str(destination), "fetch", "--quiet", "--depth=1", "origin", source_sha],
        env=env,
    )
    _run_checkout_step(
        ["git", "-C", str(destination), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        env=env,
    )


def _module_name(path: str) -> str:
    normalized = path.replace("\\", "/")
    suffix = Path(normalized).suffix
    if suffix:
        normalized = normalized[: -len(suffix)]
    if normalized.endswith("/__init__"):
        normalized = normalized[: -len("/__init__")]
    return normalized.strip("/").replace("/", ".")


def _safe_relative_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file() or ".git" in path.parts:
            continue
        try:
            relative = path.relative_to(root)
            resolved = path.resolve(strict=True)
        except ValueError:
            continue
        except OSError:
            continue
        if root.resolve() not in resolved.parents:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if len(relative.parts) > 20 or size > 1_000_000:
            continue
        files.append(relative)
    return files


def _python_file_facts(path: Path, relative: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "file": relative,
            "language": "python",
            "symbols": [],
            "imports": [],
            "calls": [],
            "parse_error": f"{exc.msg} at line {exc.lineno}",
        }

    symbols = []
    imports = set()
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(
                {
                    "name": node.name,
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                }
            )
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name:
                calls.append({"name": name, "line": node.lineno})
    return {
        "file": relative,
        "language": "python",
        "symbols": symbols[:500],
        "imports": sorted(imports)[:500],
        "calls": calls[:1000],
    }


CALL_RE = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")
CALL_KEYWORDS = {
    "catch", "class", "def", "else", "except", "for", "func", "function", "if",
    "interface", "match", "new", "return", "struct", "switch", "trait", "while",
}


def _calls_from_text(text: str) -> list[dict[str, Any]]:
    calls = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for name in CALL_RE.findall(line):
            if name not in CALL_KEYWORDS:
                calls.append({"name": name, "line": line_number})
    return calls[:1000]


JS_IMPORT_RE = re.compile(
    r"(?:from\s+['\"]([^'\"]+)['\"]|require\(\s*['\"]([^'\"]+)['\"]\s*\)|import\(\s*['\"]([^'\"]+)['\"]\s*\))"
)
JS_FUNCTION_RE = re.compile(
    r"(?:function\s+([A-Za-z_$][\w$]*)|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(?)"
)
JS_CLASS_RE = re.compile(r"\bclass\s+([A-Za-z_$][\w$]*)")


def _javascript_file_facts(path: Path, relative: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    imports = [next(part for part in match if part) for match in JS_IMPORT_RE.findall(text)]
    symbols = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        function = JS_FUNCTION_RE.search(line)
        if function:
            symbols.append({"name": function.group(1) or function.group(2), "kind": "function", "line": line_number})
        class_match = JS_CLASS_RE.search(line)
        if class_match:
            symbols.append({"name": class_match.group(1), "kind": "class", "line": line_number})
    return {
        "file": relative,
        "language": "typescript" if path.suffix.lower() in {".ts", ".tsx"} else "javascript",
        "symbols": symbols[:500],
        "imports": sorted(set(imports))[:500],
        "calls": _calls_from_text(text),
    }


GO_IMPORT_RE = re.compile(r'^\s*import\s+(?:[A-Za-z_.]+\s+)?["`]([^"`]+)["`]', re.MULTILINE)
GO_IMPORT_BLOCK_RE = re.compile(r"\bimport\s*\((.*?)\)", re.DOTALL)
GO_BLOCK_PATH_RE = re.compile(r'(?:^|\s)(?:[A-Za-z_.]+\s+)?["`]([^"`]+)["`]')
GO_FUNCTION_RE = re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(")
GO_TYPE_RE = re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+(struct|interface)\b")


def _go_file_facts(path: Path, relative: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    imports = set(GO_IMPORT_RE.findall(text))
    for block in GO_IMPORT_BLOCK_RE.findall(text):
        imports.update(GO_BLOCK_PATH_RE.findall(block))
    symbols = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        function = GO_FUNCTION_RE.search(line)
        if function:
            symbols.append({"name": function.group(1), "kind": "function", "line": line_number})
        type_match = GO_TYPE_RE.search(line)
        if type_match:
            symbols.append({"name": type_match.group(1), "kind": type_match.group(2), "line": line_number})
    return {
        "file": relative,
        "language": "go",
        "symbols": symbols[:500],
        "imports": sorted(imports)[:500],
        "calls": _calls_from_text(text),
    }


JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.*]+)\s*;", re.MULTILINE)
JAVA_TYPE_RE = re.compile(r"\b(class|interface|enum|record)\s+([A-Za-z_]\w*)")
JAVA_METHOD_RE = re.compile(
    r"^\s*(?:public|protected|private|static|final|synchronized|abstract|native|default|\s)+"
    r"(?:<[\w, ? extends super]+>\s+)?[\w<>\[\],.?]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:throws\s+[^{]+)?\{?"
)


def _java_file_facts(path: Path, relative: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    symbols = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        type_match = JAVA_TYPE_RE.search(line)
        if type_match:
            symbols.append({"name": type_match.group(2), "kind": type_match.group(1), "line": line_number})
        method = JAVA_METHOD_RE.search(line)
        if method and method.group(1) not in CALL_KEYWORDS:
            symbols.append({"name": method.group(1), "kind": "method", "line": line_number})
    return {
        "file": relative,
        "language": "java",
        "symbols": symbols[:500],
        "imports": sorted(set(JAVA_IMPORT_RE.findall(text)))[:500],
        "calls": _calls_from_text(text),
    }


RUST_IMPORT_RE = re.compile(r"^\s*(?:use|mod)\s+([^;{]+)", re.MULTILINE)
RUST_SYMBOL_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(fn|struct|enum|trait)\s+([A-Za-z_]\w*)")


def _rust_file_facts(path: Path, relative: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    symbols = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = RUST_SYMBOL_RE.search(line)
        if match:
            symbols.append({"name": match.group(2), "kind": match.group(1), "line": line_number})
    imports = [value.strip().replace("::", ".") for value in RUST_IMPORT_RE.findall(text)]
    return {
        "file": relative,
        "language": "rust",
        "symbols": symbols[:500],
        "imports": sorted(set(imports))[:500],
        "calls": _calls_from_text(text),
    }


def _generic_file_facts(path: Path, relative: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "file": relative,
        "language": path.suffix.lstrip(".") or "unknown",
        "symbols": [],
        "imports": [],
        "calls": _calls_from_text(text),
    }


def _source_file_facts(path: Path, relative: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _python_file_facts(path, relative)
    if suffix in {".js", ".jsx", ".ts", ".tsx"}:
        return _javascript_file_facts(path, relative)
    if suffix == ".go":
        return _go_file_facts(path, relative)
    if suffix == ".java":
        return _java_file_facts(path, relative)
    if suffix == ".rs":
        return _rust_file_facts(path, relative)
    return _generic_file_facts(path, relative)


def _import_matches_changed_file(importer: str, imported: str, changed_path: str) -> bool:
    raw = imported.strip().replace("::", ".").replace("/", ".")
    raw = re.sub(r"\.(?:js|jsx|ts|tsx|py|go|java|rs)$", "", raw)
    changed_module = _module_name(changed_path)
    changed_stem = Path(changed_path).stem
    changed_parent = Path(changed_path).parent.as_posix().replace("/", ".")
    if imported.startswith("."):
        importer_parent = Path(importer).parent
        resolved = (importer_parent / imported).as_posix()
        normalized_parts = []
        for part in resolved.split("/"):
            if part == "..":
                if normalized_parts:
                    normalized_parts.pop()
            elif part not in {"", "."}:
                normalized_parts.append(part)
        raw = _module_name("/".join(normalized_parts))
    candidates = {raw.strip("."), raw.rsplit(".", 1)[-1], changed_module, changed_stem}
    return (
        raw == changed_module
        or raw.endswith(f".{changed_stem}")
        or (changed_parent and raw.endswith(changed_parent))
        or changed_module.endswith(f".{raw.rsplit('.', 1)[-1]}")
        or changed_stem in candidates and raw.rsplit(".", 1)[-1] == changed_stem
    )


def build_symbol_dependency_map(
    checkout: Path,
    changed_files: list[str],
    *,
    max_impacted_files: int,
) -> dict[str, Any]:
    """Build a compact symbol/import map and direct reverse-dependency scope."""
    facts: dict[str, dict[str, Any]] = {}
    for relative_path in _safe_relative_files(checkout):
        if relative_path.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
            continue
        relative = relative_path.as_posix()
        full_path = checkout / relative_path
        facts[relative] = _source_file_facts(full_path, relative)

    changed = [path for path in changed_files if path in facts]
    changed_symbol_names = {
        symbol["name"]
        for path in changed
        for symbol in facts[path].get("symbols", [])
        if symbol.get("name")
    }
    callers = []
    reverse_dependencies = []
    for path, file_facts in facts.items():
        if path in changed:
            continue
        imports = set(file_facts.get("imports", []))
        if any(
            _import_matches_changed_file(path, imported, changed_path)
            for imported in imports
            for changed_path in changed
        ):
            reverse_dependencies.append(path)
        matching_calls = [
            call for call in file_facts.get("calls", []) if call.get("name") in changed_symbol_names
        ]
        if matching_calls:
            callers.append({"file": path, "calls": matching_calls[:20]})

    impacted_files = []
    for path in changed + reverse_dependencies + [item["file"] for item in callers]:
        if path not in impacted_files:
            impacted_files.append(path)
        if len(impacted_files) >= max_impacted_files:
            break

    changed_symbols = [
        {**symbol, "file": path}
        for path in changed
        for symbol in facts[path].get("symbols", [])
    ][:200]
    return {
        "changed_files": changed,
        "impacted_files": impacted_files,
        "reverse_dependencies": reverse_dependencies[:max_impacted_files],
        "callers": callers[:max_impacted_files],
        "changed_symbols": changed_symbols,
        "files": {path: facts[path] for path in impacted_files if path in facts},
        "files_scanned": len(facts),
    }


def build_repository_context(checkout: Path, symbol_map: dict[str, Any], limit: int = 30_000) -> list[dict]:
    snippets = []
    used = 0
    changed = set(symbol_map.get("changed_files", []))
    for path in symbol_map.get("impacted_files", []):
        if path in changed:
            continue
        full_path = (checkout / path).resolve()
        if checkout.resolve() not in full_path.parents or not full_path.is_file():
            continue
        text = full_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        caller_lines = []
        for caller in symbol_map.get("callers", []):
            if caller.get("file") == path:
                caller_lines.extend(call.get("line") for call in caller.get("calls", []))
        selected = []
        for line in caller_lines[:5]:
            if not isinstance(line, int):
                continue
            start = max(line - 4, 0)
            end = min(line + 3, len(lines))
            selected.append({"start_line": start + 1, "content": "\n".join(lines[start:end])})
        if not selected:
            selected = [{"start_line": 1, "content": "\n".join(lines[:40])}]
        item = {"file": path, "snippets": selected}
        size = len(json.dumps(item, ensure_ascii=False))
        if used + size > limit:
            break
        snippets.append(item)
        used += size
    return snippets


def _targeted_test_paths(checkout: Path, symbol_map: dict[str, Any]) -> list[str]:
    candidates = []
    changed = symbol_map.get("changed_files", [])
    stems = {Path(path).stem.removeprefix("test_") for path in changed}
    for relative in _safe_relative_files(checkout):
        path = relative.as_posix()
        if relative.suffix != ".py":
            continue
        if relative.name.startswith("test_") and (
            path in changed or relative.stem.removeprefix("test_") in stems
        ):
            candidates.append(path)
    return candidates[:20]


def build_analysis_commands(
    checkout: Path,
    symbol_map: dict[str, Any],
    *,
    timeout_seconds: int,
    include_tests: bool,
) -> list[dict[str, Any]]:
    impacted = [
        path for path in symbol_map.get("impacted_files", []) if (checkout / path).is_file()
    ][:40]
    python_paths = [path for path in impacted if Path(path).suffix == ".py"]
    commands: list[dict[str, Any]] = []
    if python_paths:
        commands.extend(
            [
                {
                    "name": "compile",
                    "argv": ["python", "-m", "py_compile", *python_paths],
                    "timeout_seconds": min(timeout_seconds, 120),
                },
                {
                    "name": "ruff",
                    "argv": ["ruff", "check", "--output-format", "json", *python_paths],
                    "timeout_seconds": timeout_seconds,
                },
                {
                    "name": "bandit",
                    "argv": ["bandit", "-f", "json", "-q", *python_paths],
                    "timeout_seconds": timeout_seconds,
                },
            ]
        )

    source_paths = [
        path for path in impacted if Path(path).suffix.lower() in SUPPORTED_SOURCE_SUFFIXES
    ]
    if source_paths:
        commands.append({
            "name": "semgrep",
            "argv": [
                "semgrep",
                "--config",
                "/app/semgrep-rules.yml",
                "--json",
                "--quiet",
                "--metrics",
                "off",
                *source_paths,
            ],
            "timeout_seconds": timeout_seconds,
        })

    if include_tests and (checkout / "tests").is_dir():
        targeted_tests = _targeted_test_paths(checkout, symbol_map)
        test_targets = targeted_tests or ["tests"]
        commands.append(
            {
                "name": "tests",
                "argv": [
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--disable-warnings",
                    *test_targets,
                ],
                "timeout_seconds": timeout_seconds,
            }
        )
    return commands


def _run_local_commands(checkout: Path, commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    safe_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": tempfile.gettempdir(),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    for command in commands:
        started = time.perf_counter()
        try:
            response = subprocess.run(
                command["argv"],
                cwd=checkout,
                env=safe_env,
                capture_output=True,
                text=True,
                timeout=int(command["timeout_seconds"]),
                check=False,
            )
            results.append(
                {
                    "name": command["name"],
                    "argv": command["argv"],
                    "exit_code": response.returncode,
                    "stdout": response.stdout[-MAX_COMMAND_OUTPUT:],
                    "stderr": response.stderr[-MAX_COMMAND_OUTPUT:],
                    "timed_out": False,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
        except FileNotFoundError as exc:
            results.append(
                {
                    "name": command["name"],
                    "argv": command["argv"],
                    "exit_code": 127,
                    "stdout": "",
                    "stderr": str(exc),
                    "timed_out": False,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
        except subprocess.TimeoutExpired as exc:
            results.append(
                {
                    "name": command["name"],
                    "argv": command["argv"],
                    "exit_code": 124,
                    "stdout": str(exc.stdout or "")[-MAX_COMMAND_OUTPUT:],
                    "stderr": str(exc.stderr or "")[-MAX_COMMAND_OUTPUT:],
                    "timed_out": True,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
    return results


def _submit_filesystem_sandbox(
    root: Path,
    workspace: Path,
    commands: list[dict[str, Any]],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    requests_dir = root / "requests"
    results_dir = root / "results"
    requests_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    relative_workspace = workspace.resolve().relative_to(root.resolve()).as_posix()
    request = {
        "id": request_id,
        "workspace": relative_workspace,
        "commands": commands,
        "created_at": time.time(),
        "deadline": time.time() + max(timeout_seconds - 5, 1),
    }
    temporary_request = requests_dir / f".{request_id}.tmp"
    request_path = requests_dir / f"{request_id}.json"
    temporary_request.write_text(json.dumps(request), encoding="utf-8")
    temporary_request.replace(request_path)

    result_path = results_dir / f"{request_id}.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            result_path.unlink(missing_ok=True)
            return payload
        time.sleep(0.25)
    request_path.unlink(missing_ok=True)
    raise TimeoutError("Timed out waiting for isolated sandbox result")


def _severity(value: str, default: str = "medium") -> str:
    normalized = str(value).lower()
    return {
        "error": "high",
        "warning": "medium",
        "info": "low",
        "high": "high",
        "medium": "medium",
        "low": "low",
    }.get(normalized, default)


def _normalize_tool_path(value: Any) -> str:
    path = str(value or "").replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    if "/repo/" in path:
        path = path.rsplit("/repo/", 1)[1]
    return path


def parse_static_findings(command_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for result in command_results:
        name = result.get("name")
        stdout = result.get("stdout") or ""
        try:
            payload = json.loads(stdout)
        except (json.JSONDecodeError, TypeError):
            continue
        if name == "ruff" and isinstance(payload, list):
            for item in payload:
                edits = ((item.get("fix") or {}).get("edits") or [])
                suggested_patch = None
                if len(edits) == 1 and isinstance(edits[0].get("content"), str):
                    suggested_patch = edits[0]["content"].rstrip("\n")
                findings.append(
                    {
                        "description": item.get("message", "Ruff finding"),
                        "file": _normalize_tool_path(item.get("filename", "")),
                        "line": (item.get("location") or {}).get("row"),
                        "severity": "medium",
                        "impact": f"Static lint rule {item.get('code', 'unknown')} matched.",
                        "confidence": 0.98,
                        "finding_type": "definite_bug",
                        "evidence": item.get("message", ""),
                        "rule_id": f"ruff.{item.get('code', 'unknown')}",
                        "evidence_source": "ruff",
                        "why_this_matters": "Resolving this lint violation prevents a concrete correctness or maintainability defect in the changed code.",
                        "suggested_patch": suggested_patch,
                    }
                )
        elif name == "bandit" and isinstance(payload, dict):
            for item in payload.get("results", []):
                findings.append(
                    {
                        "description": item.get("issue_text", "Bandit security finding"),
                        "file": _normalize_tool_path(item.get("filename", "")),
                        "line": item.get("line_number"),
                        "severity": _severity(item.get("issue_severity", "medium")),
                        "impact": f"Bandit rule {item.get('test_id', 'unknown')} matched.",
                        "confidence": {"high": 0.98, "medium": 0.85, "low": 0.65}.get(
                            str(item.get("issue_confidence", "medium")).lower(), 0.8
                        ),
                        "finding_type": "definite_bug",
                        "evidence": item.get("code", "")[:500],
                        "rule_id": f"bandit.{item.get('test_id', 'unknown')}",
                        "evidence_source": "bandit",
                        "why_this_matters": item.get("more_info") or item.get("issue_text", "This security pattern can expose production data or execution paths."),
                    }
                )
        elif name == "semgrep" and isinstance(payload, dict):
            for item in payload.get("results", []):
                extra = item.get("extra") or {}
                findings.append(
                    {
                        "description": extra.get("message", "Semgrep finding"),
                        "file": _normalize_tool_path(item.get("path", "")),
                        "line": (item.get("start") or {}).get("line"),
                        "severity": _severity(extra.get("severity", "warning")),
                        "impact": f"Semgrep rule {item.get('check_id', 'unknown')} matched.",
                        "confidence": 0.95,
                        "finding_type": "definite_bug",
                        "evidence": (extra.get("lines") or extra.get("message") or "")[:500],
                        "rule_id": f"semgrep.{item.get('check_id', 'unknown')}",
                        "evidence_source": "semgrep",
                        "why_this_matters": extra.get("metadata", {}).get("impact") or extra.get("message", "This SAST rule identifies a security or reliability risk."),
                        "suggested_patch": extra.get("fix"),
                    }
                )
    return findings[:500]


def summarize_execution(command_results: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = []
    for result in command_results:
        summaries.append(
            {
                "name": result.get("name"),
                "exit_code": result.get("exit_code"),
                "timed_out": bool(result.get("timed_out")),
                "duration_ms": result.get("duration_ms", 0),
                "output_excerpt": ((result.get("stdout") or "") + "\n" + (result.get("stderr") or ""))[-4000:],
            }
        )
    tests = next((item for item in summaries if item["name"] == "tests"), None)
    return {
        "commands": summaries,
        "tests": tests or {"name": "tests", "exit_code": None, "status": "not_run"},
        "all_tools_available": not any(item.get("exit_code") == 127 for item in summaries),
    }


def build_repository_intelligence(
    metadata: dict[str, Any],
    pr_context: dict[str, Any],
    *,
    github_token: str | None,
    config: IntelligenceConfig | None = None,
) -> dict[str, Any]:
    config = config or IntelligenceConfig.from_environment()
    if not config.enabled:
        return {"status": "disabled"}

    root = config.sandbox_root.resolve()
    workspace = root / "workspaces" / uuid.uuid4().hex
    checkout = workspace / "repo"
    root.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        clone_pr_head(
            metadata["repository"],
            metadata["source_sha"],
            checkout,
            github_token,
        )
        changed_files = [file.get("filename", "") for file in pr_context.get("files", [])]
        symbol_map = build_symbol_dependency_map(
            checkout,
            changed_files,
            max_impacted_files=config.max_impacted_files,
        )
        repository_context = build_repository_context(checkout, symbol_map)
        include_tests = config.mode == "filesystem" or config.allow_local_test_execution
        commands = build_analysis_commands(
            checkout,
            symbol_map,
            timeout_seconds=config.command_timeout_seconds,
            include_tests=include_tests,
        )
        if config.mode == "filesystem":
            sandbox_result = _submit_filesystem_sandbox(
                root,
                workspace,
                commands,
                timeout_seconds=config.request_timeout_seconds,
            )
            if sandbox_result.get("error"):
                raise RuntimeError(f"Isolated sandbox failed: {sandbox_result['error']}")
            command_results = sandbox_result.get("results", [])
            isolation = sandbox_result.get("isolation", {})
            if not isolation.get("network_disabled"):
                raise RuntimeError("Isolated sandbox did not confirm network isolation")
        else:
            static_commands = [command for command in commands if command["name"] != "tests"]
            if config.allow_local_test_execution:
                static_commands = commands
            command_results = _run_local_commands(checkout, static_commands)
            isolation = {
                "backend": "local",
                "network_disabled": False,
                "test_execution_allowed": config.allow_local_test_execution,
            }

        static_findings = parse_static_findings(command_results)
        if config.mode == "filesystem" and any(
            result.get("exit_code") == 127 for result in command_results
        ):
            raise RuntimeError("A required analysis tool is missing from the isolated sandbox")
        changed_set = set(changed_files)
        changed_findings = [finding for finding in static_findings if finding.get("file") in changed_set]
        contextual_findings = [finding for finding in static_findings if finding.get("file") not in changed_set]
        return {
            "status": "completed",
            "source_sha": metadata["source_sha"],
            "review_scope": {
                "changed_files": symbol_map.get("changed_files", []),
                "impacted_files": symbol_map.get("impacted_files", []),
                "reverse_dependencies": symbol_map.get("reverse_dependencies", []),
                "callers": symbol_map.get("callers", []),
            },
            "symbol_map": {
                "changed_symbols": symbol_map.get("changed_symbols", []),
                "files": symbol_map.get("files", {}),
                "files_scanned": symbol_map.get("files_scanned", 0),
            },
            "repository_context": repository_context,
            "static_findings": changed_findings,
            "contextual_static_findings": contextual_findings[:100],
            "execution": summarize_execution(command_results),
            "isolation": isolation,
        }
    finally:
        if not config.retain_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
