import json
import sys
from pathlib import Path

import pytest

from taskchamber.cli import main
from taskchamber.config import (
    CommandDocumentSourceConfig,
    load_configuration,
    load_project_policy,
)
from taskchamber.core.contracts import TaskKind


def _load(path: Path):
    configuration = load_configuration(environment={}, working_directory=path.parent)
    return load_project_policy(
        configuration,
        working_directory=path.parent,
        config_file=path,
    )


def test_project_toml_loads_task_ceiling_and_parameterized_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "taskchamber.toml"
    config.write_text(
        f"""
schema_version = 1

[policy]
allowed_capabilities = ["workspace.read", "workspace.search", "documents.read"]
default_capabilities = ["workspace.read"]
max_document_sources = 4

[policy.workspace]
root = "."
include = ["src/**/*.py", "tests/**/*.py"]
exclude = ["tests/private/**"]
max_requested_paths = 8

[tasks.research]
allowed_capabilities = ["workspace.read", "workspace.search", "documents.read"]
default_capabilities = ["workspace.read"]
max_turns = 10

[tasks.summarize]
allowed_capabilities = ["workspace.read"]
default_capabilities = ["workspace.read"]

[tasks.review]
allowed_capabilities = ["workspace.read", "workspace.search"]
default_capabilities = ["workspace.read", "workspace.search"]

[document_sources.record_detail]
kind = "command"
description = "Read one record"
aliases = ["record detail", "记录详情"]
executable = {json.dumps(sys.executable)}
args = ["-c", "print('ok')", "{{record_id}}"]
env_refs = ["RECORDS_API_TOKEN"]
output_format = "json_document"

[document_sources.record_detail.parameters.record_id]
description = "Record identifier"
aliases = ["record", "记录编号"]
pattern = "^rec_[A-Za-z0-9_-]+$"
example = "rec_demo"
""",
        encoding="utf-8",
    )

    loaded = _load(config)

    assert loaded.policy.max_document_sources == 4
    assert loaded.workspace_root == tmp_path.resolve()
    assert loaded.policy.workspace.include == ("src/**/*.py", "tests/**/*.py")
    assert loaded.policy.tasks[TaskKind.REVIEW].allowed == (
        "workspace.read",
        "workspace.search",
    )
    source = loaded.document_sources["record_detail"]
    assert isinstance(source, CommandDocumentSourceConfig)
    assert source.argv[-1] == "{record_id}"
    assert source.env_allow == ("RECORDS_API_TOKEN",)
    assert source.output_format == "json_document"
    assert source.parameters[0].example == "rec_demo"

    main(["policy", "show", "--config", str(config)])
    shown = capsys.readouterr().out
    assert '"record_detail"' in shown
    assert "RECORDS_API_TOKEN" not in shown
    assert sys.executable not in shown
    assert '"executable"' not in shown


def test_project_toml_rejects_unknown_or_broadening_fields(tmp_path: Path) -> None:
    config = tmp_path / "taskchamber.toml"
    config.write_text(
        """
[policy]
allowed_capabilities = ["workspace.read"]
unknown_switch = true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        _load(config)

    config.write_text(
        """
[policy]
allowed_capabilities = ["workspace.read"]

[tasks.review]
allowed_capabilities = ["workspace.read", "workspace.search"]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exceeds the project policy"):
        _load(config)


def test_management_cli_initializes_edits_and_validates_policy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "taskchamber.toml"
    main(["config", "init", "--path", str(config)])
    assert config.is_file()
    template = config.read_text(encoding="utf-8")
    assert '# output_format = "json_document"' in template
    assert '# document_id = "record.json"' in template

    main(
        [
            "policy",
            "deny",
            "review",
            "workspace.search",
            "--config",
            str(config),
        ]
    )
    assert "workspace.search" not in _load(config).policy.tasks[TaskKind.REVIEW].allowed

    main(
        [
            "policy",
            "allow",
            "review",
            "workspace.search",
            "--config",
            str(config),
        ]
    )
    main(
        [
            "policy",
            "set-default",
            "review",
            "workspace.read",
            "workspace.search",
            "--config",
            str(config),
        ]
    )
    loaded = _load(config)
    assert loaded.policy.tasks[TaskKind.REVIEW].defaults == (
        "workspace.read",
        "workspace.search",
    )

    main(["policy", "validate", "--config", str(config)])
    assert "Valid TaskChamber policy" in capsys.readouterr().out
