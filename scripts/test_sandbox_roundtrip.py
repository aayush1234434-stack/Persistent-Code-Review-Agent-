"""Exercise the real shared-volume sandbox and prove it has no network interface."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from review_intelligence import _submit_filesystem_sandbox


def main() -> int:
    root = Path(os.environ.get("SANDBOX_ROOT", "/sandbox")).resolve()
    workspace = root / "workspaces" / f"integration-{uuid.uuid4().hex}"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    try:
        program = (
            "from pathlib import Path\n"
            "interfaces=[line.split(':',1)[0].strip() for line in Path('/proc/net/dev').read_text().splitlines() if ':' in line]\n"
            "external=[name for name in interfaces if name != 'lo']\n"
            "assert not external, f'unexpected network interfaces: {external}'\n"
            "Path('sandbox-proof.txt').write_text('isolated')\n"
            "print('network-isolated')\n"
        )
        payload = _submit_filesystem_sandbox(
            root,
            workspace,
            [{
                "name": "network-isolation-proof",
                "argv": ["python", "-c", program],
                "timeout_seconds": 20,
            }],
            timeout_seconds=45,
        )
        assert not payload.get("error"), payload
        assert payload.get("isolation", {}).get("network_disabled") is True
        assert payload.get("isolation", {}).get("resource_limits") is True
        result = payload["results"][0]
        assert result["exit_code"] == 0, result
        assert "network-isolated" in result["stdout"]
        assert (repo / "sandbox-proof.txt").read_text() == "isolated"
        print("PASS: sandbox round trip used a unique UID, resource limits, and loopback-only networking")
        return 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
