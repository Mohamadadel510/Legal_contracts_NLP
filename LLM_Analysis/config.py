"""
Central configuration for the contract-AI engine.

Everything here is overridable via environment variables so the same code
runs unchanged in a notebook, a local dev run, and the deployed pipeline
(e.g. different Ollama host per environment, different model per stage).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


@dataclass(frozen=True)
class Settings:
    # Ollama connection. OLLAMA_HOST is read natively by the `ollama` client,
    # but we surface it here too so pipeline code can log/validate it.
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Separate models per stage, since drafting used a lighter model (7b)
    # than analysis (14b) in the original notebook. Override independently.
    analysis_model: str = os.getenv("CONTRACT_AI_ANALYSIS_MODEL", "qwen2.5:14b")
    drafting_model: str = os.getenv("CONTRACT_AI_DRAFTING_MODEL", "qwen2.5:7b")

    analysis_temperature: float = _env_float("CONTRACT_AI_ANALYSIS_TEMP", 0.1)
    summary_temperature: float = _env_float("CONTRACT_AI_SUMMARY_TEMP", 0.2)
    drafting_temperature: float = _env_float("CONTRACT_AI_DRAFTING_TEMP", 0.2)

    clause_retries: int = _env_int("CONTRACT_AI_CLAUSE_RETRIES", 2)
    # Cap on concurrent in-flight LLM calls during parallel clause analysis.
    max_concurrent_clauses: int = _env_int("CONTRACT_AI_MAX_CONCURRENCY", 4)

    request_timeout_s: float = _env_float("CONTRACT_AI_TIMEOUT_S", 120.0)


settings = Settings()
