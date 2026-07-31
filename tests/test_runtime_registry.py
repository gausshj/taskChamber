import os
import subprocess
import sys
from pathlib import Path

import pytest

from taskchamber.application import (
    create_default_service,
    create_runtime_from_environment,
    create_runtime_registry,
)
from taskchamber.config import load_configuration
from taskchamber.isolation import NoSandbox
from taskchamber.runtimes.claude import ClaudeAgentSdkRuntime
from taskchamber.runtimes.fake import FakeRuntime
from taskchamber.runtimes.registry import RuntimeFactoryContext, RuntimeRegistry


def _context(tmp_path: Path) -> RuntimeFactoryContext:
    return RuntimeFactoryContext(
        configuration=load_configuration(environment={}, working_directory=tmp_path),
        sandbox=NoSandbox(),
    )


def test_registry_constructs_a_registered_runtime(tmp_path: Path) -> None:
    registry = RuntimeRegistry()
    registry.register("custom", lambda _context: FakeRuntime())

    runtime = registry.create("custom", _context(tmp_path))

    assert isinstance(runtime, FakeRuntime)
    assert registry.names == ("custom",)


def test_registry_rejects_duplicate_and_unknown_runtimes(tmp_path: Path) -> None:
    registry = RuntimeRegistry()

    def initial_factory(_context: RuntimeFactoryContext) -> FakeRuntime:
        return FakeRuntime()

    def duplicate_factory(_context: RuntimeFactoryContext) -> FakeRuntime:
        return FakeRuntime()

    registry.register("custom", initial_factory)
    missing_context = _context(tmp_path)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("custom", duplicate_factory)
    with pytest.raises(ValueError, match="available runtimes: custom"):
        registry.create("missing", missing_context)


def test_registry_loads_an_adapter_factory_by_import_target(tmp_path: Path) -> None:
    registry = RuntimeRegistry()
    registry.register_lazy(
        "fake",
        "taskchamber.runtimes.fake.factory:create_runtime",
    )

    assert isinstance(registry.create("fake", _context(tmp_path)), FakeRuntime)


def test_registry_rejects_an_invalid_lazy_target() -> None:
    registry = RuntimeRegistry()

    with pytest.raises(ValueError, match="module:factory"):
        registry.register_lazy("broken", "missing-separator")


def test_builtin_registry_exposes_lazy_adapter_names() -> None:
    assert create_runtime_registry(discover=False).names == ("claude", "fake")


def test_registry_discovers_installed_runtime_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEntryPoint:
        name = "installed"

        @staticmethod
        def load():
            return lambda _context: FakeRuntime()

    monkeypatch.setattr(
        "taskchamber.runtimes.registry.metadata.entry_points",
        lambda **_kwargs: (FakeEntryPoint(),),
    )
    registry = RuntimeRegistry()

    registry.discover()

    assert isinstance(registry.create("installed", _context(tmp_path)), FakeRuntime)


def test_composition_selects_fake_runtime_without_a_provider_sdk(tmp_path: Path) -> None:
    runtime = create_runtime_from_environment(
        environment={"TASKCHAMBER_RUNTIME": "fake"},
        working_directory=tmp_path,
    )
    assert isinstance(runtime, FakeRuntime)


def test_default_service_reads_non_secret_settings_from_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "TASKCHAMBER_DEFAULT_PROFILE=custom_profile\n",
        encoding="utf-8",
    )

    service = create_default_service(
        environment={"TASKCHAMBER_RUNTIME": "fake"},
        working_directory=tmp_path,
    )

    assert service.settings.default_profile == "custom_profile"


def test_claude_factory_receives_custom_dotenv_provider_profiles(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "TASKCHAMBER_PROFILES=custom",
                "TASKCHAMBER_PROFILE__CUSTOM__RUNTIME=claude",
                "TASKCHAMBER_PROFILE__CUSTOM__API_FORMAT=anthropic",
                "TASKCHAMBER_PROFILE__CUSTOM__BASE_URL=https://provider.example/anthropic",
                "TASKCHAMBER_PROFILE__CUSTOM__MODEL=custom-model",
                "TASKCHAMBER_PROFILE__CUSTOM__API_KEY=test-token",
            ]
        ),
        encoding="utf-8",
    )

    runtime = create_runtime_from_environment(
        environment={"TASKCHAMBER_RUNTIME": "claude"},
        working_directory=tmp_path,
        registry=create_runtime_registry(discover=False),
    )

    assert isinstance(runtime, ClaudeAgentSdkRuntime)
    profile = runtime._provider_for("custom")
    assert profile.model == "custom-model"
    assert profile.base_url == "https://provider.example/anthropic"
    assert (
        runtime.environment_for(profile, config_dir=tmp_path / "config")["ANTHROPIC_AUTH_TOKEN"]
        == "test-token"
    )


def test_importing_transport_does_not_import_claude_sdk(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = "\n".join(
        [
            "import builtins",
            "from pathlib import Path",
            "real_import = builtins.__import__",
            "def blocked(name, *args, **kwargs):",
            "    if name == 'claude_agent_sdk' or name.startswith('claude_agent_sdk.'):",
            "        raise AssertionError('vendor SDK imported by MCP core')",
            "    return real_import(name, *args, **kwargs)",
            "builtins.__import__ = blocked",
            "import taskchamber.transport.mcp",
            "from taskchamber.application import create_runtime_from_environment",
            "runtime = create_runtime_from_environment(",
            "    environment={'TASKCHAMBER_RUNTIME': 'fake'},",
            f"    working_directory=Path({str(tmp_path)!r}),",
            ")",
            "assert runtime.name == 'fake'",
        ]
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(repository_root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
