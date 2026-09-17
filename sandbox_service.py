"""Network-isolated command executor for pull-request workspaces."""

from __future__ import annotations

import contextlib
import json
import os
import resource
import signal
import stat
import subprocess
import time
from pathlib import Path
from typing import Any


SANDBOX_ROOT = Path(os.environ.get("SANDBOX_ROOT", "/sandbox")).resolve()
REQUESTS_DIR = SANDBOX_ROOT / "requests"
RESULTS_DIR = SANDBOX_ROOT / "results"
WORKSPACES_DIR = SANDBOX_ROOT / "workspaces"
RUNNING_DIR = SANDBOX_ROOT / "running"
HEALTH_PATH = SANDBOX_ROOT / "health.json"
MAX_OUTPUT = 200_000
CHILD_UID_BASE = int(os.environ.get("SANDBOX_CHILD_UID_BASE", "100000"))


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def _inside(parent: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_command(command: Any) -> dict[str, Any]:
    if not isinstance(command, dict):
        raise ValueError("Sandbox command must be an object")
    argv = command.get("argv")
    if not isinstance(argv, list) or not argv or len(argv) > 100:
        raise ValueError("Sandbox command argv is invalid")
    if not all(isinstance(arg, str) and "\x00" not in arg and len(arg) <= 4096 for arg in argv):
        raise ValueError("Sandbox command contains an invalid argument")
    timeout = int(command.get("timeout_seconds", 300))
    if timeout < 1 or timeout > 1800:
        raise ValueError("Sandbox command timeout is invalid")
    return {
        "name": str(command.get("name", "command"))[:80],
        "argv": argv,
        "timeout_seconds": timeout,
    }


def _drop_privileges_and_limit(uid: int, gid: int) -> None:
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)


def _make_workspace_available(workspace: Path, uid: int, gid: int) -> None:
    for path in [workspace, *workspace.rglob("*")]:
        if path.is_symlink():
            continue
        original_mode = path.stat().st_mode
        os.chown(path, uid, gid)
        if path.is_dir():
            mode = 0o700
        else:
            mode = 0o700 if original_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) else 0o600
        os.chmod(path, mode)


def _run_command(
    workspace: Path,
    command: dict[str, Any],
    *,
    uid: int,
    gid: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    sandbox_home = workspace / ".sandbox-home"
    sandbox_tmp = workspace / ".sandbox-tmp"
    for directory in (sandbox_home, sandbox_tmp):
        directory.mkdir(exist_ok=True)
        os.chown(directory, uid, gid)
        os.chmod(directory, 0o700)
    safe_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(sandbox_home),
        "TMPDIR": str(sandbox_tmp),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "NO_COLOR": "1",
        "SEMGREP_SEND_METRICS": "off",
    }
    try:
        process = subprocess.Popen(
            command["argv"],
            cwd=workspace / "repo",
            env=safe_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=lambda: _drop_privileges_and_limit(uid, gid),
        )
        try:
            stdout, stderr = process.communicate(timeout=command["timeout_seconds"])
            timed_out = False
            exit_code = process.returncode
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            timed_out = True
            exit_code = 124
    except FileNotFoundError as exc:
        stdout, stderr, timed_out, exit_code = "", str(exc), False, 127
    return {
        "name": command["name"],
        "argv": command["argv"],
        "exit_code": exit_code,
        "stdout": stdout[-MAX_OUTPUT:],
        "stderr": stderr[-MAX_OUTPUT:],
        "timed_out": timed_out,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _process_request(request_path: Path) -> None:
    running_path = RUNNING_DIR / request_path.name
    try:
        request_path.replace(running_path)
    except FileNotFoundError:
        return
    request_id = running_path.stem
    try:
        payload = json.loads(running_path.read_text(encoding="utf-8"))
        if payload.get("id") != request_id or not request_id.isalnum():
            raise ValueError("Sandbox request identity is invalid")
        workspace = (SANDBOX_ROOT / Path(str(payload.get("workspace", "")))).resolve()
        if not _inside(WORKSPACES_DIR, workspace) or not (workspace / "repo").is_dir():
            raise ValueError("Sandbox workspace is outside the allowed root")
        commands = [_validate_command(command) for command in payload.get("commands", [])]
        child_uid = CHILD_UID_BASE + (int(request_id[:8], 16) % 50_000)
        child_gid = child_uid
        _make_workspace_available(workspace, child_uid, child_gid)
        command_results = []
        deadline = float(payload.get("deadline", time.time() + 300))
        for command in commands:
            remaining = int(deadline - time.time())
            if remaining < 1:
                command_results.append({
                    "name": command["name"],
                    "argv": command["argv"],
                    "exit_code": 124,
                    "stdout": "",
                    "stderr": "Global sandbox request deadline reached",
                    "timed_out": True,
                    "duration_ms": 0,
                })
                continue
            command["timeout_seconds"] = min(command["timeout_seconds"], remaining)
            command_results.append(
                _run_command(workspace, command, uid=child_uid, gid=child_gid)
            )
        result = {
            "id": request_id,
            "results": command_results,
            "isolation": {
                "backend": "container-filesystem",
                "network_disabled": os.environ.get("SANDBOX_NETWORK_DISABLED") == "true",
                "child_uid": child_uid,
                "resource_limits": True,
            },
        }
    except Exception as exc:
        result = {
            "id": request_id,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "results": [],
            "isolation": {"backend": "container-filesystem", "failed": True},
        }
    finally:
        running_path.unlink(missing_ok=True)
    _write_json_atomic(RESULTS_DIR / f"{request_id}.json", result)


def serve() -> None:
    for directory in (REQUESTS_DIR, RESULTS_DIR, WORKSPACES_DIR, RUNNING_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for stale_request in RUNNING_DIR.glob("*.json"):
        stale_request.replace(REQUESTS_DIR / stale_request.name)
    while True:
        _write_json_atomic(HEALTH_PATH, {"status": "healthy", "updated_at": time.time()})
        for request_path in sorted(REQUESTS_DIR.glob("*.json")):
            _process_request(request_path)
        time.sleep(0.2)


if __name__ == "__main__":
    serve()
