import json
from pathlib import Path

import pytest

from taskchamber.config import (
    CommandDocumentSourceConfig,
    DirectoryDocumentSourceConfig,
    ProviderProfile,
    load_configuration,
    load_document_source_configs,
    load_provider_profiles,
)


def test_dotenv_adds_a_custom_provider_without_mutating_process_env(tmp_path: Path) -> None:
    secret = "not-a-real-secret"
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "TASKCHAMBER_PROFILES=custom_provider",
                "TASKCHAMBER_PROFILE__CUSTOM_PROVIDER__RUNTIME=claude",
                "TASKCHAMBER_PROFILE__CUSTOM_PROVIDER__API_FORMAT=anthropic",
                "TASKCHAMBER_PROFILE__CUSTOM_PROVIDER__BASE_URL=https://example.invalid/anthropic",
                "TASKCHAMBER_PROFILE__CUSTOM_PROVIDER__MODEL=custom-test",
                "TASKCHAMBER_PROFILE__CUSTOM_PROVIDER__API_KEY_FIELD=auth_token",
                f"TASKCHAMBER_PROFILE__CUSTOM_PROVIDER__API_KEY={secret}",
            ]
        ),
        encoding="utf-8",
    )

    configuration = load_configuration(environment={}, working_directory=tmp_path)
    profiles = load_provider_profiles(
        configuration,
        defaults={},
        default_runtime="claude",
    )

    profile = profiles["custom_provider"]
    assert profile.runtime == "claude"
    assert profile.api_format == "anthropic"
    assert profile.base_url == "https://example.invalid/anthropic"
    assert profile.model == "custom-test"
    assert configuration.secrets.get(profile.credential_ref) == secret
    assert secret not in repr(configuration)
    assert secret not in repr(profile)


def test_process_environment_overrides_dotenv_profile_and_secret(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "TASKCHAMBER_PROFILES=custom",
                "TASKCHAMBER_PROFILE__CUSTOM__API_FORMAT=anthropic",
                "TASKCHAMBER_PROFILE__CUSTOM__MODEL=dotenv-model",
                "TASKCHAMBER_PROFILE__CUSTOM__API_KEY=dotenv-secret",
            ]
        ),
        encoding="utf-8",
    )
    environment = {
        "TASKCHAMBER_PROFILE__CUSTOM__MODEL": "process-model",
        "TASKCHAMBER_PROFILE__CUSTOM__API_KEY": "process-secret",
    }

    configuration = load_configuration(
        environment=environment,
        working_directory=tmp_path,
    )
    profile = load_provider_profiles(
        configuration,
        defaults={},
        default_runtime="claude",
    )["custom"]

    assert profile.model == "process-model"
    assert configuration.secrets.get(profile.credential_ref) == "process-secret"


def test_dotenv_does_not_interpolate_unrelated_parent_secrets(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "TASKCHAMBER_PROFILES=custom",
                "TASKCHAMBER_PROFILE__CUSTOM__API_FORMAT=anthropic",
                "TASKCHAMBER_PROFILE__CUSTOM__MODEL=model",
                "TASKCHAMBER_PROFILE__CUSTOM__API_KEY=${HOST_ONLY_SECRET}",
            ]
        ),
        encoding="utf-8",
    )
    configuration = load_configuration(
        environment={"HOST_ONLY_SECRET": "parent-secret"},
        working_directory=tmp_path,
    )
    profile = load_provider_profiles(
        configuration,
        defaults={},
        default_runtime="claude",
    )["custom"]

    assert configuration.secrets.get(profile.credential_ref) == "${HOST_ONLY_SECRET}"


def test_dynamic_profile_can_override_a_builtin_without_repeating_every_field(
    tmp_path: Path,
) -> None:
    default = ProviderProfile(
        name="glm",
        runtime="claude",
        api_format="anthropic",
        base_url="https://default.invalid/anthropic",
        model="default-model",
        credential_ref="Z_AI_API_KEY",
        api_key_field="auth_token",
    )
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "TASKCHAMBER_PROFILES=glm",
                "TASKCHAMBER_PROFILE__GLM__MODEL=custom-model",
                "TASKCHAMBER_PROFILE__GLM__API_KEY=profile-secret",
            ]
        ),
        encoding="utf-8",
    )

    configuration = load_configuration(environment={}, working_directory=tmp_path)
    profile = load_provider_profiles(
        configuration,
        defaults={"glm": default},
        default_runtime="claude",
    )["glm"]

    assert profile.model == "custom-model"
    assert profile.base_url == default.base_url
    assert profile.credential_ref == "TASKCHAMBER_PROFILE__GLM__API_KEY"


def test_explicit_missing_env_file_fails_without_showing_secret_data(tmp_path: Path) -> None:
    missing = tmp_path / "missing.env"
    with pytest.raises(ValueError, match="does not exist") as error:
        load_configuration(environment={}, env_file=missing)
    assert "API_KEY" not in str(error.value)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://remote.example/api",
        "https://user:secret@example.invalid/api",
        "https://example.invalid/api?token=secret",
    ],
)
def test_provider_profile_rejects_unsafe_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError):
        ProviderProfile(
            name="unsafe",
            runtime="claude",
            api_format="anthropic",
            base_url=base_url,
            model="model",
            credential_ref="SECRET_REF",
        )


def test_provider_profile_allows_local_http_endpoint() -> None:
    profile = ProviderProfile(
        name="local",
        runtime="claude",
        api_format="anthropic",
        base_url="http://127.0.0.1:8080/anthropic",
        model="model",
        credential_ref="LOCAL_KEY",
    )
    assert profile.base_url == "http://127.0.0.1:8080/anthropic"


def test_provider_profile_accepts_cc_switch_style_api_key_field() -> None:
    profile = ProviderProfile(
        name="compatible",
        runtime="claude",
        api_format="anthropic",
        base_url="https://example.invalid/anthropic",
        model="model",
        credential_ref="COMPATIBLE_KEY",
        api_key_field="auth-token",
    )
    assert profile.api_key_field == "auth_token"


def test_document_sources_load_from_dotenv_with_relative_directory_and_fixed_argv(
    tmp_path: Path,
) -> None:
    argv = json.dumps(["doc-cli", "fetch", "--query", "{query}"])
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "TASKCHAMBER_DOCUMENT_SOURCES=docs,cli_docs",
                "TASKCHAMBER_DOCUMENT_SOURCE__DOCS__KIND=directory",
                "TASKCHAMBER_DOCUMENT_SOURCE__DOCS__ROOT=../shared-docs",
                "TASKCHAMBER_DOCUMENT_SOURCE__DOCS__INCLUDE=**/*.md,**/*.txt",
                "TASKCHAMBER_DOCUMENT_SOURCE__CLI_DOCS__KIND=command",
                f"TASKCHAMBER_DOCUMENT_SOURCE__CLI_DOCS__ARGV={argv}",
                "TASKCHAMBER_DOCUMENT_SOURCE__CLI_DOCS__OUTPUT_FORMAT=json",
                "TASKCHAMBER_DOCUMENT_SOURCE__CLI_DOCS__ENV_ALLOW=DOC_API_TOKEN",
            ]
        ),
        encoding="utf-8",
    )
    configuration = load_configuration(environment={}, working_directory=tmp_path)

    sources = load_document_source_configs(configuration, base_directory=tmp_path)

    assert isinstance(sources["docs"], DirectoryDocumentSourceConfig)
    assert sources["docs"].root == (tmp_path / "../shared-docs").resolve()
    assert sources["docs"].include == ("**/*.md", "**/*.txt")
    assert isinstance(sources["cli_docs"], CommandDocumentSourceConfig)
    assert sources["cli_docs"].argv == ("doc-cli", "fetch", "--query", "{query}")
    assert sources["cli_docs"].env_allow == ("DOC_API_TOKEN",)


def test_document_command_configuration_rejects_shell_string_and_unknown_placeholder(
    tmp_path: Path,
) -> None:
    base = {
        "TASKCHAMBER_DOCUMENT_SOURCES": "cli_docs",
        "TASKCHAMBER_DOCUMENT_SOURCE__CLI_DOCS__KIND": "command",
        "TASKCHAMBER_DOCUMENT_SOURCE__CLI_DOCS__ARGV": "doc-cli fetch; touch marker",
    }
    configuration = load_configuration(environment=base, working_directory=tmp_path)
    with pytest.raises(ValueError, match="JSON argv array"):
        load_document_source_configs(configuration, base_directory=tmp_path)

    base["TASKCHAMBER_DOCUMENT_SOURCE__CLI_DOCS__ARGV"] = json.dumps(["doc-cli", "{unsafe}"])
    configuration = load_configuration(environment=base, working_directory=tmp_path)
    with pytest.raises(ValueError, match="only supports"):
        load_document_source_configs(configuration, base_directory=tmp_path)

    base["TASKCHAMBER_DOCUMENT_SOURCE__CLI_DOCS__ARGV"] = json.dumps(["doc-cli", "--query={query}"])
    configuration = load_configuration(environment=base, working_directory=tmp_path)
    with pytest.raises(ValueError, match="standalone argv item"):
        load_document_source_configs(configuration, base_directory=tmp_path)

    base["TASKCHAMBER_DOCUMENT_SOURCE__CLI_DOCS__ARGV"] = json.dumps(
        ["/bin/sh", "-c", "fixed", "{query}"]
    )
    configuration = load_configuration(environment=base, working_directory=tmp_path)
    with pytest.raises(ValueError, match="command shell"):
        load_document_source_configs(configuration, base_directory=tmp_path)
