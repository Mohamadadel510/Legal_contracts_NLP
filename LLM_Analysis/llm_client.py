"""
Thin wrapper around the `ollama` client.

Centralizing the call here means: one retry policy, one place to change
providers later (e.g. swap Ollama for a hosted endpoint), and one place to
add timing/logging instrumentation for the whole pipeline.
"""
from __future__ import annotations

import logging
from typing import Type, TypeVar

try:
    from ollama import chat as ollama_chat
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    ollama_chat = None

from pydantic import BaseModel

from .config import settings

logger = logging.getLogger("contract_ai.llm_client")

T = TypeVar("T", bound=BaseModel)


class LLMCallError(RuntimeError):
    """Raised when a structured LLM call fails after all retries."""

    def __init__(self, context: str, cause: Exception):
        super().__init__(f"LLM call failed for {context}: {cause}")
        self.context = context
        self.cause = cause


def check_model_available(model_name: str) -> bool:
    """Health-check helper: call this at service startup, not per-request."""
    try:
        chat(model=model_name, messages=[{"role": "user", "content": "ping"}])
        return True
    except Exception as exc:  # noqa: BLE001 - we want to report any failure
        logger.warning("Model '%s' unreachable: %s", model_name, exc)
        return False


def call_structured(
    *,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    response_schema: Type[T],
    temperature: float,
    retries: int = 0,
    context: str = "request",
) -> T:
    """Call the LLM and parse the response into `response_schema`.

    Raises LLMCallError if every attempt fails.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = chat(
                model=model,
                messages=messages,
                format=response_schema.model_json_schema(),
                options={"temperature": temperature},
            )
            return response_schema.model_validate_json(response.message.content)
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised typed
            last_exc = exc
            logger.warning(
                "Attempt %d/%d failed for %s: %s", attempt + 1, retries + 1, context, exc
            )

    raise LLMCallError(context, last_exc)  # type: ignore[arg-type]
