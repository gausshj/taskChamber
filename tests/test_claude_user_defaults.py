import json
from pathlib import Path

import pytest

from taskchamber.application import create_default_service, create_runtime_registry
from taskchamber.config import load_configuration
from taskchamber.core.contracts import ExecutionPolicy, TaskKind, TaskRequest
from taskchamber.runtimes.claude import ClaudeAgentSdkRuntime
from taskchamber.runtimes.claude.profiles import (
    CLAUDE_CODE_DEFAULT_CREDENTIAL,
    CLAUDE_CODE_DEFAULT_MODEL,
    CLAUDE_CODE_DEFAULT_PROFILE,
)
from taskchamber.runtimes.claude.user_defaults import (
    SETTINGS_FILE_VARIABLE,
    load_claude_code_defaults,
)


def _write_settings(path: Path, *, token: str = "settings-token") -> None:
    path.write_text(
        json.dumps(
            {
                "model": "settings-model",
                "enabledPlugins": {"must-not-load": True},
                "hooks": {"PreToolUse": [{"command": "must-not-run"}]},
                "env": {
                    "ANTHROPIC_AUTH_TOKEN": token,
                    "ANTHROPIC_BASE_URL": "https://provider.example/api/anthropic",
                    "ANTHROPIC_MODEL": "fallback-model",
                    "UNRELATED_SECRET": "must-not-forward",
                },
            }
        ),
        encoding="utf-8",
    )


def test_loader_extracts_only_claude_code_provider_routing(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    _write_settings(settings_file)
    configuration = load_configuration(
        environment={SETTINGS_FILE_VARIABLE: str(settings_file)},
        working_directory=tmp_path,
    )

    defaults = load_claude_code_defaults(configuration)

    assert defaults.profile.name == CLAUDE_CODE_DEFAULT_PROFILE
    assert defaults.profile.base_url == "https://provider.example/api/anthropic"
    assert defaults.profile.model == "settings-model"
    assert defaults.profile.api_key_field == "auth_token"
    assert defaults.secrets.get(CLAUDE_CODE_DEFAULT_CREDENTIAL) == "settings-token"
    assert "settings-token" not in repr(defaults)


def test_loader_returns_an_unavailable_profile_when_settings_are_absent(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    configuration = load_configuration(
        environment={SETTINGS_FILE_VARIABLE: str(missing)},
        working_directory=tmp_path,
    )

    defaults = load_claude_code_defaults(configuration)

    assert defaults.profile.model == CLAUDE_CODE_DEFAULT_MODEL
    assert defaults.secrets.get(CLAUDE_CODE_DEFAULT_CREDENTIAL) is None


def test_invalid_claude_code_settings_fail_without_leaking_values(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"env":{"ANTHROPIC_AUTH_TOKEN":"secret"}', encoding="utf-8")
    configuration = load_configuration(
        environment={SETTINGS_FILE_VARIABLE: str(settings_file)},
        working_directory=tmp_path,
    )

    with pytest.raises(ValueError, match="not valid JSON") as error:
        load_claude_code_defaults(configuration)

    assert "secret" not in str(error.value)


def test_claude_factory_falls_back_to_sanitized_settings_defaults(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    _write_settings(settings_file)

    service = create_default_service(
        environment={
            "TASKCHAMBER_RUNTIME": "claude",
            SETTINGS_FILE_VARIABLE: str(settings_file),
        },
        working_directory=tmp_path,
        registry=create_runtime_registry(discover=False),
    )

    assert service.settings.default_profile == CLAUDE_CODE_DEFAULT_PROFILE
    assert isinstance(service.runtime, ClaudeAgentSdkRuntime)
    profile = service.runtime._provider_for(CLAUDE_CODE_DEFAULT_PROFILE)
    options = service.runtime.build_options(
        _request(CLAUDE_CODE_DEFAULT_PROFILE),
        _policy(tmp_path),
        config_dir=tmp_path / "config",
    )
    assert profile.model == "settings-model"
    assert options.model == "settings-model"
    assert options.setting_sources == []
    assert options.env["ANTHROPIC_AUTH_TOKEN"] == "settings-token"
    assert "UNRELATED_SECRET" not in options.env


def test_one_project_profile_takes_precedence_over_claude_code_defaults(
    tmp_path: Path,
) -> None:
    settings_file = tmp_path / "settings.json"
    _write_settings(settings_file, token="user-default-token")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "TASKCHAMBER_PROFILES=project_provider",
                "TASKCHAMBER_PROFILE__PROJECT_PROVIDER__RUNTIME=claude",
                "TASKCHAMBER_PROFILE__PROJECT_PROVIDER__API_FORMAT=anthropic",
                "TASKCHAMBER_PROFILE__PROJECT_PROVIDER__BASE_URL=https://project.example/anthropic",
                "TASKCHAMBER_PROFILE__PROJECT_PROVIDER__MODEL=project-model",
                "TASKCHAMBER_PROFILE__PROJECT_PROVIDER__API_KEY=project-token",
            ]
        ),
        encoding="utf-8",
    )

    service = create_default_service(
        environment={
            "TASKCHAMBER_RUNTIME": "claude",
            SETTINGS_FILE_VARIABLE: str(settings_file),
        },
        working_directory=tmp_path,
        registry=create_runtime_registry(discover=False),
    )

    assert service.settings.default_profile == "project_provider"
    assert isinstance(service.runtime, ClaudeAgentSdkRuntime)
    profile = service.runtime._provider_for("project_provider")
    environment = service.runtime.environment_for(profile, config_dir=tmp_path / "config")
    assert environment["ANTHROPIC_AUTH_TOKEN"] == "project-token"
    assert environment["ANTHROPIC_BASE_URL"] == "https://project.example/anthropic"


def _request(provider: str) -> TaskRequest:
    return TaskRequest(
        run_id="run-defaults",
        kind=TaskKind.RESEARCH,
        prompt="Inspect the workspace.",
        provider=provider,
        max_turns=1,
    )


def _policy(workspace_root: Path) -> ExecutionPolicy:
    return ExecutionPolicy(
        workspace_root=workspace_root,
        allowed_paths=(workspace_root,),
        system_prompt="Read only.",
        allowed_tools=("Read", "Glob", "Grep"),
        disallowed_tools=("Bash", "Edit", "Write"),
        max_turns=1,
        max_budget_usd=0.5,
        timeout_seconds=1.0,
        max_output_chars=1_000,
        max_file_bytes=1_000,
    )
