"""
File-based helpers for local testing and the CLI script only.

The real pipeline (pipeline.py) never touches disk — OCR and RAG hand it
data directly. These loaders exist so you can run/debug the engine standalone
against sample JSON files, same as the notebook did.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger("contract_ai.io_utils")

_MOCK_METADATA = {
    "contract_type": "عقد إيجار مسكن",
    "parties": "المؤجر: أحمد محمود | المستأجر: مينا سامي",
}
_MOCK_CLAUSES = [
    {
        "clause_id": 1,
        "clause_label": "البند الأول - مدة العقد والقيمة الإيجارية",
        "clause_text": "مدة العقد سنة واحدة تبدأ من 1/1/2026 وتجدد تلقائياً، والإيجار 5000 جنيه شهرياً.",
    },
    {
        "clause_id": 2,
        "clause_label": "البند الثاني - الإخلاء والشرط الجزائي",
        "clause_text": "إذا تأخر المستأجر يومين عن الدفع، يحق للمؤجر طرده فوراً وتغيير الكوالين دون حكم قضائي مع شرط جزائي 100,000 جنيه.",
    },
    {
        "clause_id": 3,
        "clause_label": "البند الثالث - الإعفاء من الضمان والصيانة",
        "clause_text": "يتنازل المستأجر عن حق مطالبة المؤجر بأي صيانة للعين أو ضمان العيوب الخفية مهما كانت جسامتها.",
    },
]
_MOCK_RAG_CONTEXT = """
- المادة (571) من القانون المدني المصري: يلتزم المؤجر بصيانة العين المؤجرة لتبقى صالحة للاستيفاء بالنفع المقصود.
- المادة (577) من القانون المدني المصري: يضمن المؤجر للمستأجر العيوب التي تحول دون الانتفاع، ويكون باطلاً كل اتفاق يعفي المؤجر من ضمان العيوب إذا تعمد إخفاءها.
- المادة (224) من القانون المدني المصري: يجوز للقاضي تخفيض الشرط الجزائي المبالغ فيه.
- قواعد آمرة: لا يجوز الإخلاء القسري للعين المؤجرة دون حكم قضائي واجب النفاذ.
"""


def load_ocr_output(file_path: str = "ocr_output.json") -> Tuple[dict, list]:
    path = Path(file_path)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Loaded OCR output from %s", file_path)
        return data.get("metadata", {}), data.get("clauses", [])
    logger.warning("%s not found, using mock data", file_path)
    return _MOCK_METADATA, _MOCK_CLAUSES


def load_rag_context(file_path: str = "rag_output.json") -> str:
    path = Path(file_path)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Loaded RAG output from %s", file_path)
        if isinstance(data, list):
            return "\n".join(f"- {item.get('text', item)}" for item in data)
        if isinstance(data, dict):
            return data.get("context_text", str(data))
    logger.warning("%s not found, using mock data", file_path)
    return _MOCK_RAG_CONTEXT


def save_json(data: dict, file_path: str) -> None:
    Path(file_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved output to %s", file_path)
