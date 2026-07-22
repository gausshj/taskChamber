import os
import subprocess
import sys
from pathlib import Path


def test_manual_stdio_script_runs_with_fake_runtime(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "manual_stdio_test.py"),
            "--runtime",
            "fake",
            "--workspace",
            str(tmp_path),
            "--question",
            "Smoke test",
        ],
        cwd=repository_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "MCP tools: research, review, summarize" in result.stdout
    assert '"status": "success"' in result.stdout
    assert '"provider": "fake"' in result.stdout
