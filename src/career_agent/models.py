from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def model_from_env(
    role: str,
    *,
    temperature: float = 0,
) -> ChatOpenAI:
    """Create a model by reading ``{role}_MODEL``, ``{role}_API_KEY``, ``{role}_BASE_URL``.

    This lets each agent role (``ORCHESTRATOR``, ``RESEARCHER``, ``WRITER`` …)
    point to a different model / provider without sharing a single global config.
    """
    prefix = role.upper()
    try:
        model = os.environ[f"{prefix}_MODEL"]
        api_key = os.environ[f"{prefix}_API_KEY"]
        base_url = os.environ[f"{prefix}_BASE_URL"]
    except KeyError as exc:
        missing = exc.args[0]
        raise SystemExit(
            f"Missing environment variable: {missing}\n"
            f"Set it in .env or via export. See .env.example for reference."
        ) from None
    return openai_compatible_model(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def openai_compatible_model(
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0,
) -> ChatOpenAI:
    """Create the initial provider adapter.

    Provider-specific behavior such as DeepSeek reasoning replay belongs in a
    dedicated adapter, not in domain agents.
    """

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=120,
        max_retries=2,
    )
