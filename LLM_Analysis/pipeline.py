"""
Public entrypoint for the pipeline. This is the only module the rest of the
system (OCR stage, RAG stage, API layer) should need to import.

Design choice: everything here takes/returns plain dicts and Pydantic
objects — no files, no display() calls. Persistence and rendering are the
caller's job (see io_utils.py for optional file-based helpers used in local
testing / the CLI script).
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Union

from .analysis import run_parallel_analysis
from .schemas import ContractUserInput, FullContractDocument

logger = logging.getLogger("contract_ai.pipeline")


def _rag_context_to_text(rag_context: Union[str, list, dict]) -> str:
    """Normalize whatever shape the RAG component hands back into the plain
    text block the prompts expect. Accepts:
      - a ready-made string
      - a list of {"text": ...} chunks (typical retriever output)
      - a dict with a "context_text" key
    """
    if isinstance(rag_context, str):
        return rag_context
    if isinstance(rag_context, list):
        return "\n".join(f"- {item.get('text', item)}" if isinstance(item, dict) else f"- {item}" for item in rag_context)
    if isinstance(rag_context, dict):
        return rag_context.get("context_text", str(rag_context))
    raise TypeError(f"Unsupported rag_context type: {type(rag_context)!r}")


class ContractPipeline:
    """Stateless facade over the analysis and drafting engines.

    Example — wiring OCR -> RAG -> this pipeline:

        ocr_result = ocr_module.extract(uploaded_file)      # {"metadata": ..., "clauses": [...]}
        rag_result = rag_module.retrieve(ocr_result)         # list[{"text": ...}] or str

        pipeline = ContractPipeline()
        report = await pipeline.analyze_contract(
            metadata=ocr_result["metadata"],
            clauses=ocr_result["clauses"],
            rag_context=rag_result,
        )
    """

    async def analyze_contract(
        self, metadata: dict, clauses: List[dict], rag_context: Union[str, list, dict]
    ) -> dict:
        """Runs the risk-analysis stage. Returns
        {"summary": OverallSummarySchema, "clauses_analysis": [ClauseRiskAnalysis, ...]}.
        """
        context_text = _rag_context_to_text(rag_context)
        return await run_parallel_analysis(metadata, clauses, context_text)

    # Convenience sync wrapper for callers that aren't already in an event loop
    # (e.g. a plain script or a sync web framework view).
    def analyze_contract_sync(
        self, metadata: dict, clauses: List[dict], rag_context: Union[str, list, dict]
    ) -> dict:
        return asyncio.run(self.analyze_contract(metadata, clauses, rag_context))
