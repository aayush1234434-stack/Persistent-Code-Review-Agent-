from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerignore_excludes_secrets_and_local_artifacts():
    dockerignore = (ROOT / ".dockerignore").read_text()
    required_patterns = {
        ".env",
        ".env.*",
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
    }
    for pattern in required_patterns:
        assert pattern in dockerignore, f".dockerignore must exclude {pattern}"
