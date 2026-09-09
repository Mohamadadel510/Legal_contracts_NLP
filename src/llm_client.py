"""
Thin wrapper around the LLM providers.

Centralizing the call here means: one retry policy, one place to change
providers later, and one place to add timing/logging instrumentation for the
whole pipeline.

Two backends are supported and pick themselves via settings.provider:

  groq    hosted. Structured output through JSON mode plus the schema in the
          system prompt, validated by Pydantic on the way back.
  ollama  local. Structured output through the native `format=<schema>`
          argument, which Ollama enforces during decoding.

Both funnel through call_structured(), so callers never see the difference.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Type, TypeVar

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


# A spent daily quota looks like any other failure to the caller, but it means
# something completely different: nothing will succeed until the quota resets,
# and the fix is to switch models, not to retry. Worth telling apart so the UI
# can say so instead of reporting a dozen mysterious clause failures.
#
# Only the per-day limit counts. A per-minute one carries the same 429 and the
# same "rate_limit_exceeded" code but clears on its own, and calling that a
# spent quota would send the reader off changing models for nothing.
_DAILY_QUOTA_MARKERS = ("requests per day", "(RPD)")


def is_quota_error(exc: BaseException) -> bool:
    text = str(exc)
    return "429" in text and any(marker in text for marker in _DAILY_QUOTA_MARKERS)


# --------------------------------------------------------------------------
# Groq backend
# --------------------------------------------------------------------------
_groq_client = None

# The free tier meters tokens per minute, and a rate-limited response carries
# the wait Groq wants. The SDK honours that header; a hand-rolled backoff does
# not, so let the SDK own rate-limit retries and keep this module's retry loop
# for schema failures only.
GROQ_SDK_RETRIES = 5


class TokenBudget:
    """Client-side tokens-per-minute throttle.

    Reacting to 429s is not enough on its own: a twelve-clause contract fired
    72 of them, and two clauses exhausted their retries and were lost. Pacing
    the requests instead keeps the run inside the budget from the start, so
    nothing is dropped. The run takes about as long either way - the limit is
    the limit - but it finishes complete.

    Usage is tracked over a rolling 60-second window, and reserved up front
    from an estimate, then corrected once the response reports what it really
    cost.
    """

    WINDOW_SECONDS = 60.0

    def __init__(self, tokens_per_minute: int, safety: float = 0.92):
        self.limit = max(1, int(tokens_per_minute * safety))
        self._events: list[tuple[float, int]] = []
        self._lock = threading.Lock()

    def _used(self, now: float) -> int:
        cutoff = now - self.WINDOW_SECONDS
        self._events = [(t, n) for t, n in self._events if t > cutoff]
        return sum(n for _, n in self._events)

    def reserve(self, estimated_tokens: int) -> float:
        """Block until `estimated_tokens` fit, then book them. Returns waited seconds."""
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                used = self._used(now)
                if used + estimated_tokens <= self.limit or not self._events:
                    self._events.append((now, estimated_tokens))
                    return waited

                # Sleep until enough of the oldest reservations have aged out
                # of the window to fit this one - not just until the single
                # oldest expires, which wakes up only to sleep again.
                need = used + estimated_tokens - self.limit
                freed = 0
                deadline = self._events[0][0]
                for timestamp, amount in self._events:
                    freed += amount
                    deadline = timestamp
                    if freed >= need:
                        break
                sleep_for = max(0.1, deadline + self.WINDOW_SECONDS - now)
            time.sleep(sleep_for)
            waited += sleep_for

    def settle(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Replace the estimate with what the call actually cost."""
        with self._lock:
            delta = actual_tokens - estimated_tokens
            if delta:
                self._events.append((time.monotonic(), delta))


# Calibrated against real clause requests rather than guessed: a 4,750-char
# Arabic prompt measured 1,670 prompt tokens (2.84 chars/token) and came back
# with 785 completion tokens. The first guess of 2.0 chars/token over-reserved
# by a third, which throttled the run to three clauses a minute instead of
# five. settle() corrects each reservation once the real usage is known, so
# these only need to be close.
_CHARS_PER_TOKEN = 2.8
_ESTIMATED_COMPLETION_TOKENS = 850

_budget = TokenBudget(settings.tokens_per_minute)

# Groq meters output tokens separately and far more tightly, and it checks the
# request's *declared* ceiling rather than what comes back. Sending no
# max_tokens means the model's own maximum - 2048 - is what gets checked, and
# the request is refused outright ("Request too large ... Requested 2048")
# even though a clause analysis really returns about 785 tokens. Declaring a
# realistic ceiling is what makes the request admissible.
_output_budget = (
    TokenBudget(settings.output_tokens_per_minute)
    if settings.output_tokens_per_minute > 0 else None
)


def _get_groq_client():
    """One client for the process. Groq's SDK is safe to share across threads."""
    global _groq_client
    if _groq_client is None:
        from groq import Groq  # imported lazily so ollama-only setups need no groq
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY غير موجود. ضعه في ملف .env أو غيّر "
                "CONTRACT_AI_PROVIDER إلى ollama."
            )
        _groq_client = Groq(api_key=settings.groq_api_key, max_retries=GROQ_SDK_RETRIES)
    return _groq_client


_JSON_INSTRUCTION = (
    "Respond with a single valid JSON object and nothing else - no prose, no "
    "markdown fences. It must conform to this JSON Schema:\n{schema}\n"
    "Arabic field values stay in Arabic. Enum fields must use one of the "
    "listed values verbatim."
)


def _compact_schema(model_cls: Type[T]) -> dict:
    """The schema minus its prose.

    Pydantic emits every Field(description=...) into the schema, which for
    ClauseRiskAnalysis is ~900 tokens of Arabic on every single request. The
    system prompt already states the task in Arabic, so the descriptions are
    a second copy of it - and on a tokens-per-minute budget that copy is what
    decides how many clauses fit in a minute. Field names, types and enum
    values are what the model actually needs to produce valid JSON.
    """
    schema = model_cls.model_json_schema()

    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in ("description", "title")}
        if isinstance(node, list):
            return [strip(item) for item in node]
        return node

    return strip(schema)


def _call_groq(
    *, model: str, system_prompt: str | None, user_prompt: str,
    response_schema: Type[T], temperature: float,
) -> T:
    schema = json.dumps(_compact_schema(response_schema), ensure_ascii=False)
    system = "\n\n".join(
        part for part in (system_prompt, _JSON_INSTRUCTION.format(schema=schema)) if part
    )

    estimate = int(
        (len(system) + len(user_prompt)) / _CHARS_PER_TOKEN
    ) + _ESTIMATED_COMPLETION_TOKENS

    waited = _budget.reserve(estimate)
    if _output_budget is not None:
        waited += _output_budget.reserve(settings.max_output_tokens)
    if waited:
        logger.info("Paced %.1fs to stay inside the token budget", waited)

    response = _get_groq_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=settings.max_output_tokens,
        response_format={"type": "json_object"},
    )

    usage = getattr(response, "usage", None)
    if usage is not None:
        _budget.settle(estimate, usage.total_tokens)
        if _output_budget is not None:
            _output_budget.settle(settings.max_output_tokens, usage.completion_tokens)

    return response_schema.model_validate_json(response.choices[0].message.content)


# --------------------------------------------------------------------------
# Ollama backend
# --------------------------------------------------------------------------
def _call_ollama(
    *, model: str, system_prompt: str | None, user_prompt: str,
    response_schema: Type[T], temperature: float,
) -> T:
    from ollama import chat  # lazy: groq-only setups need not install it

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    response = chat(
        model=model,
        messages=messages,
        format=response_schema.model_json_schema(),
        options={"temperature": temperature},
    )
    return response_schema.model_validate_json(response.message.content)


_BACKENDS = {"groq": _call_groq, "ollama": _call_ollama}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def check_model_available(model_name: str) -> bool:
    """Health-check helper: call this at service startup, not per-request."""
    try:
        if settings.provider == "groq":
            _get_groq_client().chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        else:
            from ollama import chat
            chat(model=model_name, messages=[{"role": "user", "content": "ping"}])
        return True
    except Exception as exc:  # noqa: BLE001 - we want to report any failure
        logger.warning("Model '%s' unreachable via %s: %s", model_name, settings.provider, exc)
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
    backend = _BACKENDS.get(settings.provider)
    if backend is None:
        raise LLMCallError(
            context, ValueError(f"Unknown provider: {settings.provider!r}")
        )

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return backend(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=response_schema,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised typed
            last_exc = exc
            logger.warning(
                "Attempt %d/%d failed for %s: %s", attempt + 1, retries + 1, context, exc
            )
            # Rate limits have already been waited out by the provider SDK, so
            # what reaches here is a malformed or schema-invalid response. A
            # short pause is enough; a long one would stall the whole run.
            if attempt < retries:
                time.sleep(1)

    raise LLMCallError(context, last_exc)  # type: ignore[arg-type]
