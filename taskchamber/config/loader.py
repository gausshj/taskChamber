"""Load provider profiles and secrets without mutating ``os.environ``."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from dotenv import dotenv_values

ENV_FILE_VARIABLE = "TASKCHAMBER_ENV_FILE"
PROFILES_VARIABLE = "TASKCHAMBER_PROFILES"
PROFILE_PREFIX = "TASKCHAMBER_PROFILE"

_PROFILE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RUNTIME_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_API_FORMAT = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_API_KEY_FIELDS = frozenset({"api_key", "auth_token"})
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@runtime_checkable
class SecretProvider(Protocol):
    """Resolve one named secret without exposing the backing store."""

    def get(self, reference: str) -> str | None:
        """Return a secret value or ``None`` when it is absent."""

        ...


@dataclass(frozen=True)
class MappingSecretProvider:
    """A secret provider backed by a mapping whose repr never shows values."""

    _values: Mapping[str, str | None] = field(repr=False)

    def get(self, reference: str) -> str | None:
        value = self._values.get(reference)
        if value is None or not value.strip():
            return None
        return value


@dataclass(frozen=True)
class LayeredSecretProvider:
    """Resolve secrets from highest-priority provider to lowest-priority provider."""

    _providers: tuple[SecretProvider, ...] = field(repr=False)

    def get(self, reference: str) -> str | None:
        for provider in self._providers:
            value = provider.get(reference)
            if value is not None:
                return value
        return None


@dataclass(frozen=True)
class ConfigurationView:
    """Read non-secret settings from ordered mappings without changing the process."""

    _layers: tuple[Mapping[str, str | None], ...] = field(repr=False)

    def get(self, name: str, default: str | None = None) -> str | None:
        for layer in self._layers:
            value = layer.get(name)
            if value is not None:
                return value
        return default

    def contains(self, name: str) -> bool:
        return any(name in layer and layer[name] is not None for layer in self._layers)


@dataclass(frozen=True)
class ConfigurationBundle:
    """Configuration and secrets collected from process env plus an optional dotenv file."""

    values: ConfigurationView
    secrets: SecretProvider = field(repr=False)
    env_file: Path | None = None


@dataclass(frozen=True)
class ProviderProfile:
    """A provider endpoint bound to one compatible agent runtime and API format."""

    name: str
    runtime: str
    api_format: str
    model: str
    credential_ref: str
    base_url: str | None = None
    api_key_field: str = "auth_token"

    def __post_init__(self) -> None:
        name = self.name.strip().lower()
        runtime = self.runtime.strip().lower()
        api_format = self.api_format.strip().lower()
        model = self.model.strip()
        credential_ref = self.credential_ref.strip()
        base_url = self.base_url.strip().rstrip("/") if self.base_url else None
        api_key_field = self.api_key_field.strip().lower().replace("-", "_")

        if not _PROFILE_NAME.fullmatch(name):
            raise ValueError(
                "provider profile name must use lowercase letters, digits, and underscores"
            )
        if not _RUNTIME_NAME.fullmatch(runtime):
            raise ValueError(f"provider {name!r} has an invalid runtime name")
        if not _API_FORMAT.fullmatch(api_format):
            raise ValueError(f"provider {name!r} has an invalid API format")
        if not model or len(model) > 200:
            raise ValueError(f"provider {name!r} must define a model of at most 200 characters")
        if not credential_ref or len(credential_ref) > 200:
            raise ValueError(f"provider {name!r} has an invalid credential reference")
        if api_key_field not in _API_KEY_FIELDS:
            raise ValueError(f"provider {name!r} API key field must be 'api_key' or 'auth_token'")
        if base_url is not None:
            _validate_base_url(name, base_url)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "api_format", api_format)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "credential_ref", credential_ref)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "api_key_field", api_key_field)


def load_configuration(
    *,
    environment: Mapping[str, str] | None = None,
    env_file: Path | None = None,
    working_directory: Path | None = None,
) -> ConfigurationBundle:
    """Load process settings over dotenv values without modifying ``os.environ``.

    An explicit ``TASKCHAMBER_ENV_FILE`` must exist. Without that setting, a
    ``.env`` in the server launch directory is used when present.
    """

    process_environment = environment if environment is not None else os.environ
    selected_file = _select_env_file(
        process_environment,
        explicit=env_file,
        working_directory=working_directory,
    )
    dotenv_mapping: dict[str, str | None] = {}
    if selected_file is not None:
        try:
            dotenv_mapping = dict(dotenv_values(selected_file, interpolate=False))
        except (OSError, ValueError) as exc:
            raise ValueError(f"could not read provider environment file: {selected_file}") from exc

    layers: tuple[Mapping[str, str | None], ...] = (process_environment, dotenv_mapping)
    return ConfigurationBundle(
        values=ConfigurationView(layers),
        secrets=LayeredSecretProvider(tuple(MappingSecretProvider(layer) for layer in layers)),
        env_file=selected_file,
    )


def load_provider_profiles(
    configuration: ConfigurationBundle,
    *,
    defaults: Mapping[str, ProviderProfile],
    default_runtime: str,
) -> Mapping[str, ProviderProfile]:
    """Overlay dotenv-defined profiles on adapter-owned built-in defaults."""

    profiles = dict(defaults)
    raw_names = configuration.values.get(PROFILES_VARIABLE, "") or ""
    names = _parse_profile_names(raw_names)
    for name in names:
        base = profiles.get(name)
        token = name.upper()
        prefix = f"{PROFILE_PREFIX}__{token}__"

        runtime = _setting(configuration.values, prefix + "RUNTIME", base, "runtime")
        api_format = _setting(
            configuration.values,
            prefix + "API_FORMAT",
            base,
            "api_format",
        )
        model = _setting(configuration.values, prefix + "MODEL", base, "model")
        base_url = _optional_setting(
            configuration.values,
            prefix + "BASE_URL",
            base.base_url if base else None,
        )
        api_key_field = _setting(
            configuration.values,
            prefix + "API_KEY_FIELD",
            base,
            "api_key_field",
            fallback="auth_token",
        )
        credential_key = prefix + "API_KEY"
        credential_ref = (
            credential_key
            if configuration.values.contains(credential_key) or base is None
            else base.credential_ref
        )

        profiles[name] = ProviderProfile(
            name=name,
            runtime=runtime or default_runtime,
            api_format=api_format or "",
            model=model or "",
            credential_ref=credential_ref,
            base_url=base_url,
            api_key_field=api_key_field or "auth_token",
        )

    return MappingProxyType(profiles)


def _select_env_file(
    environment: Mapping[str, str],
    *,
    explicit: Path | None,
    working_directory: Path | None,
) -> Path | None:
    configured = environment.get(ENV_FILE_VARIABLE)
    requested = explicit or (Path(configured) if configured else None)
    if requested is not None:
        path = requested.expanduser()
        if not path.is_absolute():
            path = (working_directory or Path.cwd()) / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"provider environment file does not exist: {path}")
        return path

    candidate = ((working_directory or Path.cwd()) / ".env").resolve()
    return candidate if candidate.is_file() else None


def _parse_profile_names(raw_names: str) -> tuple[str, ...]:
    names: list[str] = []
    for raw_name in raw_names.split(","):
        name = raw_name.strip().lower()
        if not name:
            continue
        if not _PROFILE_NAME.fullmatch(name):
            raise ValueError(
                "TASKCHAMBER_PROFILES must contain lowercase profile names separated by commas"
            )
        if name not in names:
            names.append(name)
    return tuple(names)


def _setting(
    values: ConfigurationView,
    key: str,
    base: ProviderProfile | None,
    attribute: str,
    *,
    fallback: str | None = None,
) -> str | None:
    value = values.get(key)
    if value is not None:
        return value.strip()
    if base is not None:
        inherited = getattr(base, attribute)
        return str(inherited) if inherited is not None else None
    return fallback


def _optional_setting(
    values: ConfigurationView,
    key: str,
    inherited: str | None,
) -> str | None:
    value = values.get(key)
    if value is None:
        return inherited
    stripped = value.strip()
    return stripped or None


def _validate_base_url(name: str, base_url: str) -> None:
    parsed = urlsplit(base_url)
    if parsed.username or parsed.password:
        raise ValueError(f"provider {name!r} base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"provider {name!r} base URL must not contain query or fragment data")
    if not parsed.hostname:
        raise ValueError(f"provider {name!r} base URL must include a hostname")
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS:
        return
    raise ValueError(f"provider {name!r} base URL must use HTTPS (HTTP is allowed for localhost)")


def secret_references(profiles: Sequence[ProviderProfile]) -> tuple[str, ...]:
    """Return stable unique credential references without resolving their values."""

    return tuple(dict.fromkeys(profile.credential_ref for profile in profiles))
