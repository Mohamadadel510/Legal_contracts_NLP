# -*- coding: utf-8 -*-
"""
Stage 1 - OCR and Arabic text preparation.

Module form of notebooks/contracts_ocr_pipeline.ipynb. The extraction,
cleaning and clause-segmentation logic is the notebook's; what changed is
everything that tied it to Colab:

  * no !apt-get / google.colab / files.upload() - extract() takes a path
  * PDFs with a real text layer go through PyMuPDF first.
  * Tesseract and Groq are optional. Missing either degrades the result,
    it does not raise.

Public API:
    extract(file_path) -> dict
"""
from __future__ import annotations

import io
import os
import re
import logging
import unicodedata
from pathlib import Path
from typing import List, Tuple
import pytesseract

# تحديد مسار التثبيت المباشر لمحرك Tesseract على ويندوز
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSERACT_CMD"] = tesseract_path
pytesseract.pytesseract.tesseract_cmd = tesseract_path

logger = logging.getLogger("contract_ai.ocr")

# --------------------------------------------------------------------------
# Optional dependencies - each stage degrades on its own
# --------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

try:
    import pytesseract
    _tess_cmd = os.environ.get("TESSERACT_CMD")
    if _tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff",
    ".docx", ".doc",
    ".txt", ".csv", ".json",
}

MIN_TEXT_LAYER_CHARS = 120
PAGE_SEPARATOR = "\n\n--- صفحة جديدة ---\n\n"


# ==========================================================================
# File type
# ==========================================================================
def detect_file_type(file_path: str) -> str:
    ext = os.path.splitext(str(file_path))[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}:
        return "image"
    if ext in {".docx", ".doc"}:
        return "word"
    if ext in {".txt", ".csv", ".json"}:
        return "text"
    return "unknown"


# ==========================================================================
# Groq refinement (LLM Contract Reconstruction)
# ==========================================================================
GROQ_TEXT_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
]

# تم تكبير سعة المقطع لضمان استيعاب البنود كاملة وعدم فصل السياق أثناء التنظيف
MAX_REFINE_CHARS = 3500

_REFINE_PROMPT = """أنت خبير في تدقيق وإعادة إعمار العقود والنصوص القانونية العربية الممسوخة بواسطة محركات الـ OCR.

المطلوب منك:
إعادة تصحيح وصياغة النص المرفق ليصبح عقداً قانونياً سليماً، واضحاً، ومقروءاً بنسبة 100% وبأعلى جودة تنسيق الماركدون (Markdown).

القواعد والتعليمات الإلزامية:
1. إجبار استعادة الصياغة القانونية: أصلح الكلمات المتقطعة والمشوهة (مثل "ال طا ار" -> "الطرف"، "ابارا بادر" -> "أبراج بدر"، "مدينال عهاور" -> "مدينة زهور"، "3032" -> "2023").
2. الهيكلة والتنسيق (Markdown):
   - استخدم العناوين العريضة للبنود: **البند الأول:**، **البند الثاني:**، وهكذا.
   - ضع أسماء الأطراف والتوقيعات في أسطر مستقلة ومنسقة.
   - استخدم القوائم الرقمية للشرط الفرعية داخل البنود.
3. حظر تحريف البيانات الرقمية أو الحساسة:
   - حافظ على الأرقام الحقيقية المذكورة في النص الأصلي (الأسعار، الأرقام القومية، المساحات، أرقام العقارات والقرارات) إذا كانت واضحة، ولا تخترع أرقاماً جديدة إطلاقاً.
   - اترك النقط أو الأقواس الفارغة (....................) للمعلومات غير الموجودة أو الممسوحة.
4. عدم الحذف أو الإيجاز: لا تلخص إطلاقاً ولا تحذف أي فقرة أو شرط قانوني مهما كان قصيراً.
5. لا تضف أي مقدمة، تعليق، أو خاتمة (مثل "إليك النص المعدل" أو "بعد المراجعة"). انطق بالعقد فوراً.

النص الممسوح ضوئياً (OCR Raw Output):
{raw_text}"""


def _groq_key() -> str:
    try:
        from .config import settings
        return settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


def groq_available() -> bool:
    return HAS_GROQ and _groq_key().startswith("gsk_")


def _split_for_refinement(text: str, limit: int = MAX_REFINE_CHARS) -> List[str]:
    """Break text into chunks on blank lines, never mid-paragraph."""
    if len(text) <= limit:
        return [text]

    chunks, current = [], ""
    for para in text.split("\n\n"):
        if current and len(current) + len(para) + 2 > limit:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def _strip_preamble(text: str) -> str:
    """تنظيف شامل لأي عبارات تقديمية من الموديل"""
    cleaned = re.sub(
        r"\A.*?(?:النص المصحح|النص المعدل|إليك العقد|العقد بعد التصحيح|كالتالي|كما يلي):\s*",
        "",
        text.strip(),
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def _refine_chunk(client, text: str) -> str:
    for model_name in GROQ_TEXT_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user",
                           "content": _REFINE_PROMPT.format(raw_text=text)}],
                temperature=0.0,
            )
            result = _strip_preamble(response.choices[0].message.content or "")
            if result and len(result) >= len(text) * 0.4:
                return result
            logger.warning("Refinement from %s looked truncated; keeping raw", model_name)
        except Exception as exc:
            logger.warning("Groq model %s failed: %s", model_name, exc)
    return text


def refine_text_with_groq(raw_text: str) -> str:
    """Repair OCR artefacts via Groq LLM."""
    if not raw_text or not raw_text.strip():
        return ""
    if not groq_available():
        logger.info("No GROQ_API_KEY - skipping LLM refinement of the OCR text")
        return raw_text

    try:
        client = Groq(api_key=_groq_key())
        return "\n\n".join(
            _refine_chunk(client, chunk) for chunk in _split_for_refinement(raw_text)
        )
    except Exception as exc:
        logger.warning("Groq refinement unavailable (%s); keeping raw text", exc)
        return raw_text


# ==========================================================================
# Image pre-processing
# ==========================================================================
def preprocess_image_for_ocr(pil_img):
    """CLAHE contrast boost before Tesseract."""
    if not HAS_CV2:
        return pil_img.convert("L")
    cv_img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(gray))


TESS_CONFIG = r"--oem 3 --psm 6 -l ara+eng"


def _ocr_image(pil_img) -> str:
    if not HAS_TESSERACT:
        raise RuntimeError(
            "Tesseract غير متاح. ثبّت tesseract-ocr مع حزمة اللغة العربية، "
            "أو اضبط متغير البيئة TESSERACT_CMD على مسار الملف التنفيذي."
        )
    return pytesseract.image_to_string(preprocess_image_for_ocr(pil_img),
                                       config=TESS_CONFIG)


# ==========================================================================
# Arabic cleaning
# ==========================================================================
_KASHIDA_AS_DAL = re.compile(r"د{3,}")
_LETTER_SMEAR = re.compile(r"([ء-ي])\1{2,}")
_DIGIT_GLUED = re.compile(r"([ء-ي])(\d)")
_WORD_GLUED = re.compile(r"(\d)([ء-ي])")

_BLOCK_START = re.compile(r"^\s*(?:البند|المادة|أولا|أولاً|ثانيا|ثانياً|ثالثا|ثالثاً|"
                          r"رابعا|رابعاً|خامسا|خامساً|\d{1,2}\s*[-.)]|\(\d{1,2}\))")
_SENTENCE_END = re.compile(r"[.:؟!]\s*$")
WRAPPED_LINE_CHARS = 60


def merge_wrapped_lines(text: str) -> str:
    """Rejoin lines the page layout broke mid-sentence."""
    lines = [ln.strip() for ln in text.split("\n")]
    out: List[str] = []

    for line in lines:
        if not line:
            continue
        if (
            out
            and len(out[-1]) < WRAPPED_LINE_CHARS * 4
            and not _SENTENCE_END.search(out[-1])
            and not _BLOCK_START.match(line)
        ):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)

    return "\n".join(out)


def repair_scan_artifacts(text: str) -> str:
    """Undo mechanical scan damage before passing to LLM."""
    if not text:
        return ""

    out = _KASHIDA_AS_DAL.sub("د", text)
    out = _LETTER_SMEAR.sub(r"\1", out)
    out = _DIGIT_GLUED.sub(r"\1 \2", out)
    out = _WORD_GLUED.sub(r"\1 \2", out)
    return merge_wrapped_lines(out)


_DAMAGE_TOKENS = re.compile(r"د{3,}|ـ{2,}")
DAMAGE_THRESHOLD = 0.06


def scan_damage_ratio(text: str) -> float:
    words = re.findall(r"[ء-ي]+", text)
    if not words:
        return 0.0

    suspect = sum(
        1 for w in words
        if len(w) == 1 or _DAMAGE_TOKENS.search(w) or re.search(r"([ء-ي])\1{2,}", w)
    )
    return suspect / len(words)


def clean_arabic_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[​-‏‪-‮]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([.،؛:!؟])\1{1,}", r"\1", text)
    return text.strip()


ORDINAL_WORDS = (
    r"(?:الأول|الأولى|الثاني|الثانية|الثالث|الثالثة|الرابع|الرابعة|الخامس|الخامسة|"
    r"السادس|السادسة|السابع|السابعة|الثامن|الثامنة|التاسع|التاسعة|العاشر|العاشرة|"
    r"الحادي\s+عشر|الثاني\s+عشر|الثالث\s+عشر|الرابع\s+عشر|الخامس\s+عشر|السادس\s+عشر|"
    r"السابع\s+عشر|الثامن\s+عشر|التاسع\s+عشر|العشرون|الثلاثون|رقم\s*\d+|\d+)"
)

STANDALONE_ORDINALS = (
    r"(?:أولاً|أولا|ثانياً|ثانيا|ثالثاً|ثالثا|رابعاً|رابعا|خامساً|خامسا|"
    r"سادساً|سادسا|سابعاً|سابعا|ثامناً|ثامنا|تاسعاً|تاسعا|عاشراً|عاشرا)"
)


def fix_ocr_clause_headers(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r'البند\s*العاشر', 'البند العاشر', text)
    text = re.sub(r'البند\s*الرا[بع]*', 'البند الرابع', text)
    text = re.sub(r'البند\s*السا[يئ]ع', 'البند السابع', text)
    text = re.sub(r'اليند', 'البند', text)

    text = re.sub(r'(البند\s+[أ-ي]+)\s*[\n\r_]+(عشر)', r'\1 \2', text)
    text = re.sub(r'(المادة\s+\w+)\s*[\n\r_]+(عشر)', r'\1 \2', text)
    text = re.sub(rf'(البند\s+{ORDINAL_WORDS})(?=[أ-ي])', r'\1 ', text)
    text = re.sub(rf'(المادة\s+{ORDINAL_WORDS})(?=[أ-ي])', r'\1 ', text)

    def clean_numeric_brackets(match):
        content = match.group(1)
        content = content.replace('"', '2').replace('؟', '2').replace("'", '')
        return f"({content.strip()})"

    text = re.sub(r'\(([^)\n]*[\d٠-٩"؟][^)\n]*)\)', clean_numeric_brackets, text)
    return text


# ==========================================================================
# Clause segmentation
# ==========================================================================
def build_advanced_clause_pattern():
    words_pattern = ORDINAL_WORDS
    ordinals_pattern = STANDALONE_ORDINALS
    patterns = [
        rf"(?:^|\n)\s*(البند\s+{words_pattern})\s*[:\-–—]?",
        rf"(?:^|\n)\s*(المادة\s+{words_pattern})\s*[:\-–—]?",
        rf"(?:^|\n)\s*({ordinals_pattern})\s*[:\-–—]?",
        r"(?:^|\n)\s*(\d{1,2}\s*[-.\)])\s*",
        r"(?:^|\n)\s*(\(\d{1,2}\))\s*",
    ]
    return re.compile("|".join(patterns), flags=re.MULTILINE)


def segment_legal_clauses(text: str) -> List[dict]:
    if not text or not text.strip():
        return []

    prepared_text = fix_ocr_clause_headers(text)
    matches = list(build_advanced_clause_pattern().finditer(prepared_text))
    clauses: List[dict] = []

    if matches:
        preamble = prepared_text[:matches[0].start()].strip()
        preamble = re.sub(r'---\s*صفحة جديدة\s*---', '', preamble).strip()
        if preamble:
            clauses.append({
                "clause_id": 1,
                "clause_label": "تمهيد / مقدمة العقد",
                "clause_text": preamble,
            })

        for idx, match in enumerate(matches):
            label = next((g for g in match.groups() if g), "").strip()
            start_pos = match.end()
            end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(prepared_text)

            clause_body = prepared_text[start_pos:end_pos].strip()
            clause_body = re.sub(r'---\s*صفحة جديدة\s*---', '', clause_body).strip()

            if clause_body:
                clauses.append({
                    "clause_id": len(clauses) + 1,
                    "clause_label": label,
                    "clause_text": clause_body,
                })
        return clauses

    paragraphs = [p.strip() for p in prepared_text.split("\n\n") if p.strip()]
    return [
        {"clause_id": i + 1, "clause_label": f"فقرة {i + 1}", "clause_text": p}
        for i, p in enumerate(paragraphs)
    ]


# ==========================================================================
# Public API
# ==========================================================================
def extract(file_path: str, refine: bool = True) -> dict:
    """Turn a contract file into clean text plus numbered clauses."""
    path = Path(file_path)
    result = {
        "status": "error",
        "metadata": {
            "file_name": path.name,
            "file_type": detect_file_type(file_path),
            "num_pages": 0,
            "method": None,
            "num_clauses": 0,
        },
        "raw_text": "",
        "clean_text": "",
        "clauses": [],
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"الملف غير موجود: {file_path}")
        return result

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        result["errors"].append(f"نوع ملف غير مدعوم: {path.suffix}")
        return result

    try:
        page_texts, method, page_count = extract_document_text(file_path, refine=refine)
    except Exception as exc:
        result["errors"].append(str(exc))
        return result

    raw_text = PAGE_SEPARATOR.join(p for p in page_texts if p and p.strip())
    if not raw_text.strip():
        result["metadata"]["method"] = method
        result["errors"].append(
            "تمت المعالجة لكن لم يُعثر على نص قابل للقراءة داخل الملف."
        )
        return result

    repaired = repair_scan_artifacts(raw_text)
    damage = scan_damage_ratio(repaired)
    result["metadata"]["damage_ratio"] = round(damage, 3)

    # إجبار التمرير والتصحيح النهائي بواسطة الـ LLM دائمًا عند وجود تفعيل للـ refine
    if refine:
        logger.info("Executing LLM legal contract reconstruction via Groq...")
        method = f"{method}+llm_refined"
        repaired = refine_text_with_groq(repaired)

    clean_text = clean_arabic_text(repaired)
    clauses = segment_legal_clauses(clean_text)

    result.update({
        "status": "ok",
        "raw_text": raw_text,
        "clean_text": clean_text,
        "clauses": clauses,
    })
    result["metadata"].update({
        "num_pages": page_count,
        "method": method,
        "num_clauses": len(clauses),
    })
    return result


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("usage: python -m modules.ocr <file>")
        raise SystemExit(1)
    print(json.dumps(extract(sys.argv[1]), ensure_ascii=False, indent=2)[:4000])
