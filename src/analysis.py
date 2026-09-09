"""
Clause-by-clause risk analysis engine.

Takes OCR output and RAG context as plain Python objects (dicts / strings) —
no file I/O here. The caller (pipeline.py, or whoever wires OCR -> RAG ->
this module) is responsible for producing those objects.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Sequence

from .config import settings
from .llm_client import call_structured
from .prompts import ANALYSIS_SYSTEM_PROMPT
from .schemas import ClauseRiskAnalysis, OverallSummarySchema

logger = logging.getLogger("contract_ai.analysis")


def _build_clause_prompt(clause: dict, metadata: dict, rag_context: str) -> str:
    return f"""
📌 **بيانات العقد:** {metadata.get('contract_type', 'عقد')}
📖 **المواد القانونية المتاحة (RAG Context):**
{rag_context}

📝 **البند المطلوب تحليله:**
- الرقم: {clause.get('clause_id')}
- العنوان: {clause.get('clause_label')}
- النص: {clause.get('clause_text')}
"""


def analyze_single_clause(
    clause: dict,
    metadata: dict,
    rag_context: str,
    *,
    model: str | None = None,
    retries: int | None = None,
) -> ClauseRiskAnalysis:
    """Synchronous single-clause analysis (used inside a worker thread)."""
    return call_structured(
        model=model or settings.analysis_model,
        system_prompt=ANALYSIS_SYSTEM_PROMPT,
        user_prompt=_build_clause_prompt(clause, metadata, rag_context),
        response_schema=ClauseRiskAnalysis,
        temperature=settings.analysis_temperature,
        retries=settings.clause_retries if retries is None else retries,
        context=f"clause_id={clause.get('clause_id')}",
    )


async def _analyze_clause_bounded(
    sem: asyncio.Semaphore, clause: dict, metadata: dict, rag_context: str
) -> ClauseRiskAnalysis:
    async with sem:
        return await asyncio.to_thread(
            analyze_single_clause, clause, metadata, rag_context
        )


def _build_summary_prompt(results: Sequence[ClauseRiskAnalysis]) -> str:
    return f"""
بناءً على نتائج البنود المحللة التالية:
{json.dumps([c.model_dump() for c in results], ensure_ascii=False)}

قم بتوليد الملخص التنفيذي وتقييم العقد الإجمالي.
"""


async def run_parallel_analysis(
    metadata: dict, clauses: List[dict], rag_context: str
) -> dict:
    """Analyze all clauses concurrently (bounded), then produce an overall
    summary. Returns {"summary": OverallSummarySchema, "clauses_analysis": [...]}.
    """
    if not clauses:
        raise ValueError("No clauses provided for analysis")

    logger.info("Analyzing %d clauses (max_concurrency=%d)", len(clauses), settings.max_concurrent_clauses)
    sem = asyncio.Semaphore(settings.max_concurrent_clauses)
    results: List[ClauseRiskAnalysis] = await asyncio.gather(
        *[_analyze_clause_bounded(sem, clause, metadata, rag_context) for clause in clauses]
    )

    summary = call_structured(
        model=settings.analysis_model,
        system_prompt=None,
        user_prompt=_build_summary_prompt(results),
        response_schema=OverallSummarySchema,
        temperature=settings.summary_temperature,
        retries=settings.clause_retries,
        context="overall_summary",
    )

    return {"summary": summary, "clauses_analysis": results}
