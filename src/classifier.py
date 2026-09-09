# -*- coding: utf-8 -*-
"""
Stage 2 - contract classification and entity extraction.

Written to fill the gap left by the missing classifier module. It is
deliberately rule-based rather than trained: with no labelled corpus at hand
a keyword model is honest about what it does, is inspectable when it gets a
contract wrong, and needs no model file to ship.

Its real job in this pipeline is narrow but load-bearing: the retriever
accepts only the nine English contract-type ids and raises ValueError on
anything else, while OCR and the analysis stage speak Arabic. This module is
the only place that translation happens.

Public API:
    classify(raw_text, clauses) -> dict
"""
from __future__ import annotations

import re
from typing import List, Optional

# --------------------------------------------------------------------------
# Contract types
# --------------------------------------------------------------------------
# `id` values are exactly the ones rag/modules/retriever.py accepts. Do not
# rename them here without changing the retriever's `supported_types` set.
CONTRACT_TYPES = [
    {
        "id": "residential",
        "ar": "عقد إيجار سكني",
        "keywords": {
            "العين المؤجرة": 4, "المؤجر": 3, "المستأجر": 3, "الأجرة": 2,
            "إيجار": 2, "القيمة الإيجارية": 3, "سكنى": 3, "شقة": 2,
            "الإخلاء": 2, "التأمين": 1,
        },
    },
    {
        "id": "agricultural",
        "ar": "عقد إيجار أرض زراعية",
        "keywords": {
            "الأرض الزراعية": 5, "أرض زراعية": 5, "مزارعة": 4, "المحصول": 3,
            "الدورة الزراعية": 4, "فدان": 3, "الري": 2, "الزراعة": 2,
        },
    },
    {
        "id": "commercial",
        "ar": "عقد إيجار محل تجاري",
        "keywords": {
            "محل تجاري": 5, "المحل التجاري": 5, "النشاط التجاري": 3,
            "الرخصة التجارية": 3, "السجل التجاري": 2, "متجر": 3,
        },
    },
    {
        "id": "employment",
        "ar": "عقد عمل",
        "keywords": {
            "صاحب العمل": 5, "العامل": 4, "الأجر الشهري": 4, "عقد العمل": 4,
            "ساعات العمل": 3, "الإجازة السنوية": 3, "التأمينات الاجتماعية": 3,
            "فترة الاختبار": 3, "الراتب": 3, "الوظيفة": 2, "الاستقالة": 2,
        },
    },
    {
        "id": "company",
        "ar": "عقد شركة",
        "keywords": {
            "رأس المال": 4, "الشركاء": 4, "الحصص": 3, "عقد الشركة": 5,
            "الأرباح والخسائر": 3, "المركز الرئيسي": 2, "المدير": 1,
        },
    },
    {
        "id": "sale",
        "ar": "عقد بيع",
        "keywords": {
            "البائع": 5, "المشتري": 5, "المبيع": 4, "الثمن": 3,
            "عقد بيع": 4, "نقل الملكية": 3, "التسليم": 1,
        },
    },
    {
        "id": "supply",
        "ar": "عقد توريد",
        "keywords": {
            "التوريد": 5, "المورد": 5, "أمر التوريد": 4, "الكميات": 2,
            "المواصفات الفنية": 2, "جهة التوريد": 3,
        },
    },
    {
        "id": "service",
        "ar": "عقد خدمات",
        "keywords": {
            "مقدم الخدمة": 5, "تقديم الخدمة": 4, "عقد خدمات": 4,
            "نطاق العمل": 2, "أتعاب": 2, "الاستشارات": 2,
        },
    },
    {
        "id": "consumer",
        "ar": "عقد استهلاكي",
        "keywords": {
            "المستهلك": 5, "حماية المستهلك": 5, "ضمان المنتج": 4,
            "الاسترجاع": 3, "المنتج": 1,
        },
    },
]

SUPPORTED_TYPE_IDS = [t["id"] for t in CONTRACT_TYPES]
AR_LABELS = {t["id"]: t["ar"] for t in CONTRACT_TYPES}

# Fallback when nothing scores. "general" is a valid bucket in the legal
# index, though retrieve() itself will not accept it as a contract_type.
UNKNOWN_TYPE = {"id": None, "ar": "نوع غير محدد"}

# The retriever rejects an unknown id outright, so an unclassifiable contract
# still has to be routed somewhere. Residential + the general fallback inside
# the retriever is the least-wrong default for Egyptian civil-code contracts.
DEFAULT_TYPE_ID = "residential"


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def _normalize(text: str) -> str:
    """Fold the letter variants that keyword matching keeps tripping over."""
    if not text:
        return ""
    text = re.sub(r"[ً-ْٰ]", "", text)  # diacritics
    text = text.replace("ـ", "")                  # tatweel
    for src in "أإآٱ":
        text = text.replace(src, "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", text)


# --------------------------------------------------------------------------
# Type detection
# --------------------------------------------------------------------------
def detect_contract_type(text: str) -> dict:
    """Score every type by weighted keyword frequency.

    Returns {type_id, type_ar, confidence, scores} where confidence is the
    winner's share of all points awarded — so a contract that matched two
    types about equally reports low confidence rather than a false certainty.
    """
    norm = _normalize(text)
    scores = {}

    for spec in CONTRACT_TYPES:
        total = 0
        for keyword, weight in spec["keywords"].items():
            hits = norm.count(_normalize(keyword))
            if hits:
                # Diminishing returns: ten mentions of "المؤجر" are not ten
                # times the evidence of one.
                total += weight * min(hits, 3)
        scores[spec["id"]] = total

    best_id = max(scores, key=scores.get)
    best_score = scores[best_id]
    total_score = sum(scores.values())

    if best_score == 0:
        return {
            "type_id": None,
            "type_ar": UNKNOWN_TYPE["ar"],
            "confidence": 0.0,
            "scores": scores,
        }

    return {
        "type_id": best_id,
        "type_ar": AR_LABELS[best_id],
        "confidence": round(best_score / total_score, 2) if total_score else 0.0,
        "scores": scores,
    }


# --------------------------------------------------------------------------
# Entity extraction
# --------------------------------------------------------------------------
_PARTY_PATTERNS = [
    # الطرف الأول (المؤجر): السيد / أحمد محمود
    r"الطرف\s+(?:الأول|الثاني|الاول)\s*\(\s*([^)]{2,25})\s*\)\s*[:：]?\s*"
    r"(?:السيد|السيدة|السادة|الشركة|شركة)?\s*/?\s*([^\n،.:]{3,60})",
    # المؤجر: أحمد محمود
    r"\b(المؤجر|المستأجر|البائع|المشتري|صاحب العمل|العامل|المورد|مقدم الخدمة)\s*"
    r"[:：]\s*(?:السيد|السيدة|السادة|الشركة|شركة)?\s*/?\s*([^\n،.:]{3,60})",
]


# A capture that contains any of these is boilerplate, not a person: the
# signature block ("الطرف الأول (المؤجر)   الطرف الثاني (المستأجر)") otherwise
# reads as a third party whose name is the label of the second one.
_NOT_A_NAME = ("الطرف", "المقيم", "(", ")", "توقيع")


def extract_parties(text: str) -> List[dict]:
    """Pull {role, name} pairs. Returns [] rather than guessing."""
    parties: List[dict] = []
    seen = set()

    for pattern in _PARTY_PATTERNS:
        for match in re.finditer(pattern, text):
            role = match.group(1).strip(" :،.")
            name = match.group(2).strip(" :،./")
            if not name or len(name) < 3:
                continue
            if any(token in name for token in _NOT_A_NAME):
                continue
            key = _normalize(name)
            if key in seen:
                continue
            seen.add(key)
            parties.append({"role": role, "name": name})

    return parties[:4]


def extract_date(text: str) -> Optional[str]:
    """First explicit date in the document — normally the contract date."""
    match = re.search(r"(\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{4})", text)
    if match:
        return re.sub(r"\s+", "", match.group(1))

    match = re.search(
        r"(\d{1,2}\s+(?:يناير|فبراير|مارس|أبريل|إبريل|مايو|يونيو|يوليو|"
        r"أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+\d{4})",
        text,
    )
    return match.group(1) if match else None


# Egyptian contracts routinely spell the amount out ("خمسة آلاف جنيه") rather
# than writing digits, so a digits-only pattern misses the headline figure on
# most real documents.
_NUMBER_WORDS = {
    _normalize(w)
    for w in (
        "واحد اثنان اثنين ثلاثة ثلاث أربعة أربع خمسة خمس ستة ست سبعة سبع "
        "ثمانية ثماني تسعة تسع عشرة عشر عشرون عشرين ثلاثون ثلاثين أربعون "
        "أربعين خمسون خمسين ستون ستين سبعون سبعين ثمانون ثمانين تسعون تسعين "
        "مائة مئة مائتان مئتان ألف الف آلاف الاف ألفاً مليون ملايين و"
    ).split()
}

_VALUE_ANCHORS = (
    "القيمة الإيجارية", "الأجر", "الأجرة", "الراتب", "الثمن", "قيمة العقد",
)

_CURRENCY = re.compile(r"جنيه(?:اً|ا)?")


def _amount_before(text: str, end: int) -> Optional[str]:
    """Read the amount sitting immediately before a currency word."""
    window = text[max(0, end - 60):end].rstrip()

    digits = re.search(r"(\d[\d,\.]*)\s*$", window)
    if digits:
        return digits.group(1)

    # Walk back over the trailing run of number words.
    words = window.split()
    picked = []
    for word in reversed(words):
        if _normalize(word.strip("،.:")) in _NUMBER_WORDS:
            picked.insert(0, word)
        else:
            break
    return " ".join(picked) if picked else None


def extract_value(text: str) -> Optional[str]:
    """Monetary value, preferring one that reads as the contract's own price."""
    matches = list(_CURRENCY.finditer(text))
    if not matches:
        return None

    # Prefer a figure introduced by a price-bearing phrase.
    for match in matches:
        context = text[max(0, match.start() - 80):match.start()]
        if any(anchor in context for anchor in _VALUE_ANCHORS):
            amount = _amount_before(text, match.start())
            if amount:
                return f"{amount} جنيه"

    for match in matches:
        amount = _amount_before(text, match.start())
        if amount:
            return f"{amount} جنيه"
    return None


def _trim_at_word(value: str, limit: int) -> str:
    """Cut to `limit` without slicing through a word or a date."""
    value = value.strip()
    if len(value) <= limit:
        return value
    cut = value[:limit]
    return cut[:cut.rfind(" ")].strip() if " " in cut else cut.strip()


def extract_duration(text: str) -> Optional[str]:
    match = re.search(
        r"مدة\s+(?:هذا\s+)?(?:العقد|الإيجار|التعاقد)\s*(?:هي)?\s*[:：]?\s*([^\n.،]{3,120})",
        text,
    )
    if match:
        return _trim_at_word(match.group(1), 60)

    match = re.search(r"(?:لمدة|مدته)\s+([^\n.،]{3,80})", text)
    return _trim_at_word(match.group(1), 40) if match else None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def classify(raw_text: str, clauses: Optional[List[dict]] = None) -> dict:
    """Identify the contract and its key entities.

    Parameters
    ----------
    raw_text : str
        Cleaned contract text (OCR stage `clean_text`).
    clauses : list[dict], optional
        Segmented clauses. Only used to widen the text searched when
        raw_text is thin.

    Returns
    -------
    dict
        contract_type      English id for the retriever, never None -
                           falls back to DEFAULT_TYPE_ID
        contract_type_ar   Arabic label for display
        type_confidence    0.0-1.0
        is_confident       False when the caller should show a warning or
                           let the user pick the type by hand
        parties, date, value, duration
        type_scores        per-type evidence, for debugging a wrong call
        status, errors
    """
    text = raw_text or ""
    if clauses:
        text = text + "\n" + "\n".join(c.get("clause_text", "") for c in clauses)

    result = {
        "contract_type": DEFAULT_TYPE_ID,
        "contract_type_ar": UNKNOWN_TYPE["ar"],
        "type_confidence": 0.0,
        "is_confident": False,
        "parties": [],
        "date": None,
        "value": None,
        "duration": None,
        "type_scores": {},
        "status": "error",
        "errors": [],
    }

    if not text.strip():
        result["errors"].append("لا يوجد نص لتحليله.")
        return result

    detection = detect_contract_type(text)
    result["type_scores"] = detection["scores"]

    if detection["type_id"] is None:
        result["errors"].append(
            "تعذر تحديد نوع العقد — لم يُعثر على أي مؤشرات معروفة. "
            f"سيتم استخدام «{AR_LABELS[DEFAULT_TYPE_ID]}» مبدئياً."
        )
    else:
        result["contract_type"] = detection["type_id"]
        result["contract_type_ar"] = detection["type_ar"]
        result["type_confidence"] = detection["confidence"]
        result["is_confident"] = detection["confidence"] >= 0.35

    result.update({
        "parties": extract_parties(text),
        "date": extract_date(text),
        "value": extract_value(text),
        "duration": extract_duration(text),
        "status": "ok",
    })
    return result


if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    if len(sys.argv) < 2:
        print("usage: python -m modules.classifier <text-file>")
        raise SystemExit(1)
    body = Path(sys.argv[1]).read_text(encoding="utf-8")
    print(json.dumps(classify(body), ensure_ascii=False, indent=2))
