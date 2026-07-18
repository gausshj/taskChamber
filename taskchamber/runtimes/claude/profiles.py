"""Built-in provider profiles understood by the Claude Agent SDK adapter."""

from ...config import ProviderProfile

CLAUDE_CODE_DEFAULT_PROFILE = "claude_code"
CLAUDE_CODE_DEFAULT_MODEL = "__claude_code_default_model__"
CLAUDE_CODE_DEFAULT_CREDENTIAL = "TASKCHAMBER_INTERNAL_CLAUDE_CODE_CREDENTIAL"

DEFAULT_PROVIDER_PROFILES: dict[str, ProviderProfile] = {
    "anthropic": ProviderProfile(
        name="anthropic",
        runtime="claude",
        api_format="anthropic",
        base_url=None,
        credential_ref="ANTHROPIC_API_KEY",
        model="claude-opus-4-8",
        api_key_field="api_key",
    ),
    "glm": ProviderProfile(
        name="glm",
        runtime="claude",
        api_format="anthropic",
        base_url="https://api.z.ai/api/anthropic",
        credential_ref="Z_AI_API_KEY",
        model="glm-5.2",
        api_key_field="auth_token",
    ),
    "deepseek": ProviderProfile(
        name="deepseek",
        runtime="claude",
        api_format="anthropic",
        base_url="https://api.deepseek.com/anthropic",
        credential_ref="DEEPSEEK_API_KEY",
        model="deepseek-v4-pro",
        api_key_field="auth_token",
    ),
}

# Compatibility aliases used by existing callers.
PROVIDERS = DEFAULT_PROVIDER_PROFILES
DEFAULT_PROVIDER = "glm"

__all__ = [
    "CLAUDE_CODE_DEFAULT_CREDENTIAL",
    "CLAUDE_CODE_DEFAULT_MODEL",
    "CLAUDE_CODE_DEFAULT_PROFILE",
    "DEFAULT_PROVIDER",
    "DEFAULT_PROVIDER_PROFILES",
    "PROVIDERS",
]
