"""Composition factory owned by the Claude Agent SDK adapter."""

from collections.abc import Mapping

from ...config import LayeredSecretProvider, load_provider_profiles
from ...core.contracts import AgentRuntime
from ..registry import RuntimeFactoryContext
from .profiles import CLAUDE_CODE_DEFAULT_PROFILE, DEFAULT_PROVIDER_PROFILES
from .user_defaults import load_claude_code_defaults


def create_runtime(context: RuntimeFactoryContext) -> AgentRuntime:
    """Create the Claude adapter while keeping its SDK an optional dependency."""

    try:
        from .runtime import ClaudeAgentSdkRuntime
    except ImportError as exc:
        raise RuntimeError(
            "Claude runtime is not installed; install the 'claude' optional dependency"
        ) from exc

    profiles = dict(
        load_provider_profiles(
            context.configuration,
            defaults=DEFAULT_PROVIDER_PROFILES,
            default_runtime="claude",
        )
    )
    user_defaults = load_claude_code_defaults(context.configuration)
    profiles.setdefault(user_defaults.profile.name, user_defaults.profile)
    return ClaudeAgentSdkRuntime(
        providers=profiles,
        secrets=LayeredSecretProvider((context.configuration.secrets, user_defaults.secrets)),
        sandbox=context.sandbox,
        configured_cli_path=context.configuration.values.get("TASKCHAMBER_CLAUDE_CLI_PATH"),
        default_profile=_default_profile(context, profiles),
    )


def _default_profile(
    context: RuntimeFactoryContext,
    profiles: Mapping[str, object],
) -> str:
    explicit = context.configuration.values.get("TASKCHAMBER_DEFAULT_PROFILE")
    if explicit:
        return explicit.strip()

    configured = context.configuration.values.get("TASKCHAMBER_PROFILES", "") or ""
    names = tuple(name.strip().lower() for name in configured.split(",") if name.strip())
    if len(names) == 1 and names[0] in profiles:
        return names[0]
    if context.configuration.secrets.get("Z_AI_API_KEY") is not None:
        return "glm"
    return CLAUDE_CODE_DEFAULT_PROFILE


__all__ = ["create_runtime"]
