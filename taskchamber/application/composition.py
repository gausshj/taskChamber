"""Application composition kept separate from MCP transport and runtime adapters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..config import (
    ConfigurationBundle,
    load_configuration,
    load_document_source_configs,
    load_project_policy,
)
from ..core.contracts import AgentRuntime
from ..core.service import ServerSettings, TaskService
from ..isolation import select_sandbox
from ..runtimes.builtins import register_builtin_runtimes
from ..runtimes.registry import RuntimeFactoryContext, RuntimeRegistry
from .documents import build_document_source_registry


def create_runtime_registry(*, discover: bool = True) -> RuntimeRegistry:
    """Register built-ins lazily, then add trusted installed entry points."""

    registry = RuntimeRegistry()
    register_builtin_runtimes(registry)
    if discover:
        registry.discover()
    return registry


def create_runtime_from_configuration(
    configuration: ConfigurationBundle,
    *,
    registry: RuntimeRegistry | None = None,
) -> AgentRuntime:
    runtime_name = configuration.values.get("TASKCHAMBER_RUNTIME", "claude") or "claude"
    sandbox_mode = configuration.values.get("TASKCHAMBER_SANDBOX", "auto") or "auto"
    context = RuntimeFactoryContext(
        configuration=configuration,
        sandbox=select_sandbox(sandbox_mode),
    )
    return (registry or create_runtime_registry()).create(runtime_name, context)


def create_runtime_from_environment(
    *,
    environment: Mapping[str, str] | None = None,
    env_file: Path | None = None,
    working_directory: Path | None = None,
    registry: RuntimeRegistry | None = None,
) -> AgentRuntime:
    configuration = load_configuration(
        environment=environment,
        env_file=env_file,
        working_directory=working_directory,
    )
    return create_runtime_from_configuration(configuration, registry=registry)


def create_default_service(
    *,
    environment: Mapping[str, str] | None = None,
    env_file: Path | None = None,
    working_directory: Path | None = None,
    registry: RuntimeRegistry | None = None,
) -> TaskService:
    launch_directory = (working_directory or Path.cwd()).expanduser().resolve()
    configuration = load_configuration(
        environment=environment,
        env_file=env_file,
        working_directory=launch_directory,
    )
    return create_service_from_configuration(
        configuration,
        working_directory=launch_directory,
        registry=registry,
    )


def create_service_from_configuration(
    configuration: ConfigurationBundle,
    *,
    working_directory: Path | None = None,
    registry: RuntimeRegistry | None = None,
) -> TaskService:
    """Build the task service from an already-loaded configuration bundle."""

    launch_directory = (working_directory or Path.cwd()).expanduser().resolve()
    loaded_policy = load_project_policy(
        configuration,
        working_directory=launch_directory,
    )
    runtime = create_runtime_from_configuration(configuration, registry=registry)
    settings = ServerSettings.from_mapping(
        configuration.values,
        default_profile=runtime.default_profile,
        default_workspace_root=loaded_policy.workspace_root,
    )
    source_configs = dict(loaded_policy.document_sources)
    source_configs.update(
        load_document_source_configs(
            configuration,
            base_directory=settings.workspace_root,
        )
    )
    return TaskService(
        runtime=runtime,
        settings=settings,
        document_sources=build_document_source_registry(source_configs, configuration),
        project_policy=loaded_policy.policy,
    )


__all__ = [
    "create_default_service",
    "create_runtime_from_configuration",
    "create_runtime_from_environment",
    "create_runtime_registry",
    "create_service_from_configuration",
]
