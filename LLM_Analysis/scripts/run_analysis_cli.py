#!/usr/bin/env python3
"""
Standalone test runner — mirrors what the notebook did manually, for
local debugging without wiring up real OCR/RAG components.

Usage:
    python scripts/run_analysis_cli.py analyze [--ocr ocr_output.json] [--rag rag_output.json]
    python scripts/run_analysis_cli.py draft
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contract_ai.io_utils import load_ocr_output, load_rag_context, save_json
from contract_ai.llm_client import check_model_available
from contract_ai.pipeline import ContractPipeline
from contract_ai.schemas import ContractUserInput
from contract_ai.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("contract_ai.cli")


async def run_analyze(args: argparse.Namespace) -> None:
    if not check_model_available(settings.analysis_model):
        logger.error(
            "Model '%s' not reachable at %s — is Ollama running? (ollama serve / ollama pull %s)",
            settings.analysis_model, settings.ollama_host, settings.analysis_model,
        )
        sys.exit(1)

    metadata, clauses = load_ocr_output(args.ocr)
    rag_context = load_rag_context(args.rag)

    pipeline = ContractPipeline()
    report = await pipeline.analyze_contract(metadata, clauses, rag_context)

    output = {
        "summary": report["summary"].model_dump(),
        "clauses_analysis": [c.model_dump() for c in report["clauses_analysis"]],
    }
    save_json(output, args.out)
    print(f"\n✅ Done. Overall risk: {report['summary'].overall_risk_score}/10")
    print(f"Saved to {args.out}")


def run_draft(args: argparse.Namespace) -> None:
    if not check_model_available(settings.drafting_model):
        logger.error("Drafting model '%s' not reachable", settings.drafting_model)
        sys.exit(1)

    user_input = ContractUserInput(
        contract_type="عقد إيجار شقة سكنية",
        party_one_name="محمود السيد علي (مؤجر)",
        party_two_name="أحمد حسن محمد (مستأجر)",
        key_terms=[
            "العين المؤجرة: شقة رقم 4 بالدور الثالث بالمعادي",
            "المدة: سنة واحدة تبدأ من 1-10-2026",
            "القيمة الإيجارية: 6000 جنيه مصري شهرياً تُدفع في بداية كل شهر",
        ],
        special_requests=[
            "إلزام المؤجر بإجراء الصيانة الأساسية والعمومية",
            "تحديد شرط جزائي متوازن يوازي إيجار شهر واحد فقط في حال الإخلال",
        ],
    )
    rag_context = """
- المادة (558) مدني مصري: الإيجار عقد يلتزم المؤجر بمقتضاه أن مكن المستأجر من الانتفاع بعين معينة لمدة محددة لقاء أجر معلوم.
- المادة (571) مدني مصري: يلتزم المؤجر بصيانة العين المؤجرة لتبقى على الحالة التي صالحة معها للاستيفاء بالنفع المقصود.
- المادة (224) مدني مصري: يجوز للقاضي تخفيض الشرط الجزائي إذا كان مبالغاً فيه.
"""
    pipeline = ContractPipeline()
    contract = pipeline.draft_contract(user_input, rag_context)
    save_json(contract.model_dump(), args.out)
    print(f"\n✅ Drafted: {contract.contract_title}")
    print(f"Saved to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Contract AI engine test runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Run clause risk analysis")
    p_analyze.add_argument("--ocr", default="ocr_output.json")
    p_analyze.add_argument("--rag", default="rag_output.json")
    p_analyze.add_argument("--out", default="contract_risk_analysis_output.json")

    p_draft = sub.add_parser("draft", help="Run contract drafting with sample input")
    p_draft.add_argument("--out", default="generated_contract.json")

    args = parser.parse_args()
    if args.command == "analyze":
        asyncio.run(run_analyze(args))
    elif args.command == "draft":
        run_draft(args)


if __name__ == "__main__":
    main()
