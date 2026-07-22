"""Reject tracked local configuration, credentials, and generated state."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

from check_distribution import FORBIDDEN_NAMES, FORBIDDEN_SUFFIXES, SECRET_PATTERNS

ALLOWED_DOTENV_FILES = {".env.example"}
FORBIDDEN_PARTS = {
    ".agents",
    ".codex",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
    "local-notes",
    "manual-test-output",
    "tmp",
}


def _tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [Path(raw.decode("utf-8")) for raw in completed.stdout.split(b"\0") if raw]


def main() -> None:
    failures: list[str] = []
    tracked_files = _tracked_files()
    for path in tracked_files:
        if not path.is_file():
            continue
        posix = PurePosixPath(path.as_posix())
        lowered_parts = tuple(part.lower() for part in posix.parts)
        basename = lowered_parts[-1]
        if any(part in FORBIDDEN_PARTS for part in lowered_parts):
            failures.append(f"generated or local state is tracked: {path}")
        if (
            basename == ".env" or basename.startswith(".env.")
        ) and basename not in ALLOWED_DOTENV_FILES:
            failures.append(f"plaintext dotenv file is tracked: {path}")
        if basename in FORBIDDEN_NAMES or posix.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"sensitive file type is tracked: {path}")

        content = path.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                failures.append(f"possible {label} in tracked file: {path}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS audited {len(tracked_files)} tracked files")


if __name__ == "__main__":
    main()
