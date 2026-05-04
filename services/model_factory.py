#!/usr/bin/env python3
"""
Model factory helpers.
"""

import os


OPENAI_COMPAT_KEY_ENV_NAMES = [
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_COMPAT_API_KEY",
    "UIBENCHKIT_API_KEY",
]
OPENAI_BASE_URL_ENV_NAMES = [
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_COMPAT_BASE_URL",
    "OPENROUTER_BASE_URL",
]
GEMINI_KEY_ENV_NAMES = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
ANTHROPIC_KEY_ENV_NAMES = ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]
QWEN_KEY_ENV_NAMES = ["QWEN_API_KEY", "DASHSCOPE_API_KEY"]


def _first_env(names):
    """Return first non-empty env value and the env var name."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip(), name
    return None, None


def _resolve_openai_base_url(default_openai_base_url: str) -> str:
    value, _ = _first_env(OPENAI_BASE_URL_ENV_NAMES)
    return value or default_openai_base_url


def get_provider_env_status(default_openai_base_url: str) -> dict:
    """Return provider-level env readiness for diagnostics/health output."""
    openai_key, openai_key_source = _first_env(OPENAI_COMPAT_KEY_ENV_NAMES)
    gemini_key, gemini_key_source = _first_env(GEMINI_KEY_ENV_NAMES)
    anthropic_key, anthropic_key_source = _first_env(ANTHROPIC_KEY_ENV_NAMES)
    qwen_key, qwen_key_source = _first_env(QWEN_KEY_ENV_NAMES)
    base_url, base_url_source = _first_env(OPENAI_BASE_URL_ENV_NAMES)

    return {
        "openai_compatible": {
            "ready": bool(openai_key),
            "key_source": openai_key_source,
            "base_url": base_url or default_openai_base_url,
            "base_url_source": base_url_source or "default",
        },
        "gemini": {
            "ready": bool(gemini_key),
            "key_source": gemini_key_source,
        },
        "anthropic": {
            "ready": bool(anthropic_key) or bool(openai_key),
            "key_source": anthropic_key_source or openai_key_source,
        },
        "qwen": {
            "ready": bool(qwen_key) or bool(openai_key),
            "key_source": qwen_key_source or openai_key_source,
        },
        "aws_bedrock": {
            "ready": bool(os.getenv("AWS_ACCESS_KEY_ID")) and bool(os.getenv("AWS_SECRET_ACCESS_KEY")),
            "region": os.getenv("AWS_REGION", "us-east-1"),
        },
    }


def create_bot_factory(*, get_model_info, supported_models: list, default_openai_base_url: str):
    """Create get_bot(model_name, user_api_key=None, user_base_url=None) factory."""
    def get_bot(model_name: str, user_api_key: str = None, user_base_url: str = None):
        family, version = get_model_info(model_name)
        openai_base_url = _resolve_openai_base_url(default_openai_base_url)
        masked_key = f"***{user_api_key[-4:]}" if user_api_key else None
        print(
            f"[get_bot] model={model_name}, family={family}, version={version}, "
            f"user_api_key={masked_key}, user_base_url={user_base_url}, openai_base_url={openai_base_url}"
        )

        if not family:
            raise ValueError(
                f"Unknown model: {model_name}. Supported families: {', '.join(supported_models)}"
            )

        if user_api_key:
            if user_base_url:
                from models import GPT4, GPT5

                if family == "gpt5":
                    return GPT5(user_api_key, model=version, base_url=user_base_url)
                return GPT4(user_api_key, model=version, base_url=user_base_url)

            if family == "gemini":
                from models import Gemini

                return Gemini(user_api_key, model=version)
            if family == "claude":
                from models import Claude

                return Claude(user_api_key, model=version)
            if family == "qwen":
                from models import QwenVL

                return QwenVL(user_api_key, model=version)
            if family == "gpt5":
                from models import GPT5

                return GPT5(user_api_key, model=version, base_url=openai_base_url)
            from models import GPT4

            return GPT4(user_api_key, model=version, base_url=openai_base_url)

        if family == "gemini":
            from models import Gemini

            api_key, api_key_source = _first_env(GEMINI_KEY_ENV_NAMES)
            if not api_key:
                raise ValueError(
                    f"Gemini API key not found. Set one of: {', '.join(GEMINI_KEY_ENV_NAMES)}"
                )
            print(f"[get_bot] Using Gemini key from {api_key_source}")
            return Gemini(api_key, model=version)

        if family == "gpt5":
            from models import GPT5

            api_key, api_key_source = _first_env(OPENAI_COMPAT_KEY_ENV_NAMES)
            if not api_key:
                raise ValueError(
                    f"OpenAI-compatible API key not found for {family}. "
                    f"Set one of: {', '.join(OPENAI_COMPAT_KEY_ENV_NAMES)}"
                )
            print(f"[get_bot] Using OpenAI-compatible key from {api_key_source}")
            return GPT5(api_key, model=version, base_url=openai_base_url)

        if family in ["gpt4", "deepseek", "grok", "doubao", "kimi"]:
            from models import GPT4

            api_key, api_key_source = _first_env(OPENAI_COMPAT_KEY_ENV_NAMES)
            if not api_key:
                raise ValueError(
                    f"OpenAI-compatible API key not found for {family}. "
                    f"Set one of: {', '.join(OPENAI_COMPAT_KEY_ENV_NAMES)}"
                )
            print(f"[get_bot] Using OpenAI-compatible key from {api_key_source}")
            return GPT4(api_key, model=version, base_url=openai_base_url)

        if family == "claude":
            from models import Claude, GPT4

            api_key, api_key_source = _first_env(ANTHROPIC_KEY_ENV_NAMES)
            if api_key:
                print(f"[get_bot] Using Anthropic key from {api_key_source}")
                return Claude(api_key, model=version)
            api_key, api_key_source = _first_env(OPENAI_COMPAT_KEY_ENV_NAMES)
            if api_key:
                print(f"[get_bot] Falling back to OpenAI-compatible key from {api_key_source} for Claude")
                return GPT4(api_key, model=version, base_url=openai_base_url)
            raise ValueError(
                f"Claude key not found. Set one of: {', '.join(ANTHROPIC_KEY_ENV_NAMES)} "
                f"or OpenAI-compatible key: {', '.join(OPENAI_COMPAT_KEY_ENV_NAMES)}"
            )

        if family == "qwen":
            from models import QwenVL, GPT4

            api_key, api_key_source = _first_env(QWEN_KEY_ENV_NAMES)
            if api_key:
                print(f"[get_bot] Using Qwen key from {api_key_source}")
                return QwenVL(api_key, model=version)
            api_key, api_key_source = _first_env(OPENAI_COMPAT_KEY_ENV_NAMES)
            if api_key:
                print(f"[get_bot] Falling back to OpenAI-compatible key from {api_key_source} for Qwen")
                return GPT4(api_key, model=version, base_url=openai_base_url)
            raise ValueError(
                f"Qwen key not found. Set one of: {', '.join(QWEN_KEY_ENV_NAMES)} "
                f"or OpenAI-compatible key: {', '.join(OPENAI_COMPAT_KEY_ENV_NAMES)}"
            )

        if family in ["mistral", "llama"]:
            from models import BedrockBot

            region = os.getenv("AWS_REGION", "us-east-1")
            print(f"Initializing {family} model ({version}) via AWS Bedrock in region {region}")
            return BedrockBot(model_id=version, region_name=region)

        raise ValueError(f"Unknown model family: {family}")

    return get_bot
