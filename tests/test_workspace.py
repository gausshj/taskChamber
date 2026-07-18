from pathlib import Path

import pytest

from taskchamber.core.capabilities import WorkspaceAccessPolicy
from taskchamber.core.policy import PolicyDeniedError
from taskchamber.core.workspace import WorkspaceSelector


def test_workspace_policy_globs_are_anchored_to_root(tmp_path: Path) -> None:
    allowed = tmp_path / "src" / "package" / "module.py"
    outside_policy = tmp_path / "other" / "src" / "x.py"
    allowed.parent.mkdir(parents=True)
    outside_policy.parent.mkdir(parents=True)
    allowed.write_text("ALLOWED = True\n", encoding="utf-8")
    outside_policy.write_text("ALLOWED = False\n", encoding="utf-8")
    selector = WorkspaceSelector(
        root=tmp_path,
        policy=WorkspaceAccessPolicy(include=("src/**/*.py",)),
        max_file_bytes=10_000,
    )

    assert selector.resolve(["src/package/module.py"], include_default=False) == (
        allowed.resolve(),
    )
    with pytest.raises(PolicyDeniedError, match="outside the project workspace policy"):
        selector.resolve(["other/src/x.py"], include_default=False)
