from pathlib import Path

import pytest

from taskchamber.core.policy import (
    PolicyDeniedError,
    WorkspaceGuard,
    staged_workspace,
)


def test_relative_file_stays_inside_workspace(tmp_path: Path) -> None:
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("print('safe')\n", encoding="utf-8")
    guard = WorkspaceGuard(tmp_path, max_file_bytes=1_000, allowed_tools=("Read",))

    assert guard.relative_file("src/example.py") == Path("src/example.py")
    assert guard.relative_file(str(source)) == Path("src/example.py")


def test_workspace_guard_rejects_parent_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    guard = WorkspaceGuard(tmp_path, max_file_bytes=1_000, allowed_tools=("Read",))

    with pytest.raises(PolicyDeniedError):
        guard.resolve_file("../outside.txt")


def test_workspace_guard_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)
    guard = WorkspaceGuard(tmp_path, max_file_bytes=1_000, allowed_tools=("Read",))

    with pytest.raises(PolicyDeniedError):
        guard.resolve_file("link.txt")


def test_tool_hook_policy_rejects_unsafe_paths(tmp_path: Path) -> None:
    guard = WorkspaceGuard(
        tmp_path,
        max_file_bytes=1_000,
        allowed_tools=("Read", "Glob", "Grep"),
    )

    with pytest.raises(PolicyDeniedError):
        guard.validate_tool_call("Read", {"file_path": "/etc/passwd"})
    with pytest.raises(PolicyDeniedError):
        guard.validate_tool_call("Glob", {"pattern": "../**/*"})
    with pytest.raises(PolicyDeniedError):
        guard.validate_tool_call("Glob", {"pattern": "~/.ssh/*"})
    with pytest.raises(PolicyDeniedError):
        guard.validate_tool_call("Bash", {"command": "pwd"})


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        ("Glob", {"pattern": ".env"}),
        ("Glob", {"pattern": "**/.env*"}),
        ("Glob", {"pattern": "**/*.pem"}),
        ("Glob", {"pattern": "secrets/**"}),
        ("Glob", {"pattern": "taskchamber.toml"}),
        ("Grep", {"pattern": "TOKEN", "glob": ".env*"}),
        ("Grep", {"pattern": "PRIVATE", "glob": "**/*.key"}),
    ],
)
def test_tool_hook_policy_rejects_protected_file_patterns(
    tmp_path: Path,
    tool_name: str,
    tool_input: dict[str, str],
) -> None:
    guard = WorkspaceGuard(
        tmp_path,
        max_file_bytes=1_000,
        allowed_tools=("Glob", "Grep"),
    )

    with pytest.raises(PolicyDeniedError, match="protected workspace path"):
        guard.validate_tool_call(tool_name, tool_input)


def test_tool_hook_policy_keeps_grep_content_patterns_distinct_from_paths(
    tmp_path: Path,
) -> None:
    guard = WorkspaceGuard(
        tmp_path,
        max_file_bytes=1_000,
        allowed_tools=("Glob", "Grep"),
    )

    guard.validate_tool_call("Glob", {"pattern": "**/*"})
    guard.validate_tool_call(
        "Grep",
        {"pattern": r"\.\./api|/v1/messages|\.env", "glob": "**/*.py"},
    )


@pytest.mark.parametrize("name", [".env", ".env.local", ".env.production", "taskchamber.toml"])
def test_workspace_guard_rejects_secret_and_policy_files(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("TOKEN=secret", encoding="utf-8")
    guard = WorkspaceGuard(tmp_path, max_file_bytes=1_000, allowed_tools=("Read",))

    with pytest.raises(PolicyDeniedError):
        guard.validate_tool_call("Read", {"file_path": name})


def test_staged_workspace_copies_a_single_file(tmp_path: Path) -> None:
    source = tmp_path / "src" / "note.md"
    source.parent.mkdir()
    source.write_text("hello", encoding="utf-8")

    with staged_workspace(
        tmp_path,
        source_paths=(source,),
        max_file_bytes=1_000,
        max_total_bytes=1_000,
        max_files=10,
    ) as snapshot:
        staged_file = snapshot.root / "src" / "note.md"
        assert snapshot.allowed_paths == (snapshot.root,)
        assert staged_file.is_file()
        assert staged_file.read_text(encoding="utf-8") == "hello"


def test_staged_workspace_filters_secret_and_protected_files(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("public", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=x", encoding="utf-8")
    (tmp_path / ".env.local").write_text("TOKEN=y", encoding="utf-8")
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "db.key").write_text("-----BEGIN-----", encoding="utf-8")

    with staged_workspace(
        tmp_path,
        source_paths=(tmp_path,),
        max_file_bytes=1_000,
        max_total_bytes=10_000,
        max_files=100,
    ) as snapshot:
        staged_names = {
            p.relative_to(snapshot.root).as_posix() for p in snapshot.root.rglob("*") if p.is_file()
        }

    assert "doc.md" in staged_names
    assert ".env" not in staged_names
    assert ".env.local" not in staged_names
    assert "secrets/db.key" not in staged_names


def test_staged_workspace_skips_symlinked_directories(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "leaked.txt").write_text("should not be staged", encoding="utf-8")
    (tmp_path / "doc.md").write_text("safe", encoding="utf-8")
    (tmp_path / "link").symlink_to(real)

    with staged_workspace(
        tmp_path,
        source_paths=(tmp_path,),
        max_file_bytes=1_000,
        max_total_bytes=10_000,
        max_files=100,
    ) as snapshot:
        staged_names = {
            p.relative_to(snapshot.root).as_posix() for p in snapshot.root.rglob("*") if p.is_file()
        }

    assert "doc.md" in staged_names
    assert "link/leaked.txt" not in staged_names


def test_staged_workspace_enforces_aggregate_limits(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 100, encoding="utf-8")

    with pytest.raises(PolicyDeniedError):
        with staged_workspace(
            tmp_path,
            source_paths=(tmp_path / "big.txt",),
            max_file_bytes=1_000,
            max_total_bytes=10,
            max_files=100,
        ):
            pass
