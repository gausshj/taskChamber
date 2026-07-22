"""Safely import provider routing fields from Claude Code settings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path

from ...config import ConfigurationBundle, MappingSecretProvider, ProviderProfile, SecretProvider
from .profiles import (
    CLAUDE_CODE_DEFAULT_CREDENTIAL,
    CLAUDE_CODE_DEFAULT_MODEL,
    CLAUDE_CODE_DEFAULT_PROFILE,
)

SETTINGS_FILE_VARIABLE = "TASKCHAMBER_CLAUDE_SETTINGS_FILE"
MAX_SETTINGS_BYTES = 1_000_000


@dataclass(frozen=True)
class ClaudeCodeDefaults:
    """One sanitized provider profile plus its separately held credential."""

    profile: ProviderProfile
    secrets: SecretProvider = field(repr=False)
    settings_file: Path


def load_claude_code_defaults(configuration: ConfigurationBundle) -> ClaudeCodeDefaults:
    """Read only provider routing fields, never plugins, hooks, tools, or permissions."""

    settings_file = _settings_file(configuration)
    data: Mapping[str, object] = {}
    if settings_file.exists():
        if not settings_file.is_file():
            raise ValueError(f"Claude Code settings path is not a file: {settings_file}")
        try:
            if settings_file.stat().st_size > MAX_SETTINGS_BYTES:
                raise ValueError("Claude Code settings file exceeds the size limit")
            parsed = json.loads(settings_file.read_text(encoding="utf-8"))
        except JSONDecodeError as exc:
            raise ValueError(
                f"Claude Code settings file is not valid JSON: {settings_file}"
            ) from exc
        except OSError as exc:
            raise ValueError(f"could not read Claude Code settings file: {settings_file}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Claude Code settings must contain a JSON object")
        data = parsed

    raw_env = data.get("env", {})
    settings_env = raw_env if isinstance(raw_env, dict) else {}
    auth_token = _string(settings_env, "ANTHROPIC_AUTH_TOKEN")
    api_key = _string(settings_env, "ANTHROPIC_API_KEY")
    credential = auth_token or api_key
    api_key_field = "auth_token" if auth_token else "api_key"
    base_url = _string(settings_env, "ANTHROPIC_BASE_URL")
    model = _string(data, "model") or _string(settings_env, "ANTHROPIC_MODEL")

    profile = ProviderProfile(
        name=CLAUDE_CODE_DEFAULT_PROFILE,
        runtime="claude",
        api_format="anthropic",
        base_url=base_url,
        model=model or CLAUDE_CODE_DEFAULT_MODEL,
        credential_ref=CLAUDE_CODE_DEFAULT_CREDENTIAL,
        api_key_field=api_key_field,
    )
    return ClaudeCodeDefaults(
        profile=profile,
        secrets=MappingSecretProvider({CLAUDE_CODE_DEFAULT_CREDENTIAL: credential}),
        settings_file=settings_file,
    )


def _settings_file(configuration: ConfigurationBundle) -> Path:
    explicit = configuration.values.get(SETTINGS_FILE_VARIABLE)
    if explicit:
        return Path(explicit).expanduser().resolve()

    config_dir = configuration.values.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return (Path(config_dir).expanduser() / "settings.json").resolve()

    home = configuration.values.get("HOME")
    home_path = Path(home).expanduser() if home else Path.home()
    return (home_path / ".claude" / "settings.json").resolve()


def _string(values: Mapping[str, object], key: str) -> str | None:
    value = values.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "MAX_SETTINGS_BYTES",
    "SETTINGS_FILE_VARIABLE",
    "ClaudeCodeDefaults",
    "load_claude_code_defaults",
]
