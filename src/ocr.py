# -*- coding: utf-8 -*-
"""
Stage 1 - OCR + Arabic document formatting (Synced with Notebook Pipeline)
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import copy
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any

logger = logging.getLogger("contract_ai.ocr")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import pytesseract
    from PIL import Image
    _tess_cmd = os.environ.get("TESSERACT_CMD")
    if _tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff",
    ".docx", ".doc",
    ".txt", ".csv", ".json",
}

# الإعدادات المتوافقة مع النوت بوك
CONFIG: Dict[str, Any] = {
    "output_dir": os.environ.get("OCR_OUTPUT_DIR", "/content/output"),
    "langs": ["ar", "eng"],
    "dpi": 300,
    "chunk_max_lines": 40,
    "chunk_overlap_lines": 3,
    "tesseract_config": r"--oem 3 --psm 6",
    "gemini_model": os.environ.get("GEMINI_OCR_MODEL", "gemini-3.6-flash"),
}

# ============================================================================
# Gemini Client Helper
# ============================================================================
def _get_gemini_api_key() -> str:
    """قراءة مفتاح API الخاص بـ Gemini من بيئة العمل."""
    return os.environ.get("GEMINI_API_KEY", "").strip()

def gemini_available() -> bool:
    return HAS_GENAI and bool(_get_gemini_api_key())

def _get_gemini_client():
    if not gemini_available():
        return None
    return genai.Client(api_key=_get_gemini_api_key())

# ============================================================================
# 1. تحميل الصفحات (PDF -> Images)
# ============================================================================
def detect_file_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}:
        return "image"
    if ext in {".docx", ".doc"}:
        return "word"
    if ext in {".txt", ".csv", ".json"}:
        return "text"
    return "unknown"

def load_pages(input_path: str, dpi: int = 300):
    """تحويل ملف الـ PDF/الصورة إلى صور PIL بوضوح 300 DPI مثل النوت بوك تماماً."""
    if not HAS_FITZ:
        raise RuntimeError("PyMuPDF (fitz) غير مثبت.")

    ext = Path(input_path).suffix.lower()
    pages = []

    if ext == ".pdf":
        with fitz.open(input_path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                pages.append(img)
    elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
        pages = [Image.open(input_path).convert("RGB")]
    else:
        raise ValueError(f"صيغة غير مدعومة: {ext}")

    if not pages:
        raise ValueError("لم يتم استخراج أي صفحة من الملف.")

    return pages

# ============================================================================
# 2. الاستخراج المحلي (Tesseract OCR)
# ============================================================================
def extract_text_local_ocr(pages) -> List[Dict]:
    """استخراج النص سطر بسطر محلياً باستخدام Tesseract."""
    if not HAS_TESSERACT:
        raise RuntimeError("Tesseract غير متاح. يرجى تثبيت pytesseract و Tesseract-OCR.")

    raw_ocr_data = []
    custom_config = CONFIG["tesseract_config"]

    for page_idx, page_image in enumerate(pages):
        page_number = page_idx + 1
        logger.info(f"جاري معالجة الصفحة {page_number} من {len(pages)} محلياً...")

        try:
            text = pytesseract.image_to_string(page_image, lang="ara+eng", config=custom_config)
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            for line_idx, line_text in enumerate(lines):
                line_id = f"p{page_number}_l{line_idx + 1}"
                raw_ocr_data.append({
                    "page": page_number,
                    "line_id": line_id,
                    "line_number": line_idx + 1,
                    "text": line_text,
                    "confidence": 1.0,
                    "bbox": None
                })
        except Exception as e:
            logger.warning(f"⚠️ خطأ في معالجة الصفحة {page_number}: {e}")

    return raw_ocr_data

# ============================================================================
# 3. حفظ البيانات الخام والنسخة المحمية (Immutable)
# ============================================================================
def save_raw_ocr(data: List[Dict], output_dir: str) -> Tuple[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw_json_path = output / "raw_ocr.json"
    with raw_json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    raw_txt_path = output / "raw_ocr.txt"
    lines_by_page: Dict[int, List[Dict]] = {}
    for item in data:
        lines_by_page.setdefault(item["page"], []).append(item)

    with raw_txt_path.open("w", encoding="utf-8") as f:
        for page in sorted(lines_by_page.keys()):
            f.write(f"===== صفحة {page} =====\n")
            for item in sorted(lines_by_page[page], key=lambda x: x["line_number"]):
                f.write((item["text"] or "") + "\n")
            f.write("\n")

    return str(raw_json_path), str(raw_txt_path)

# ============================================================================
# 4. المعالجة والتنسيق عبر Gemini صفحة بصفحة
# ============================================================================
def clean_and_format_by_page(ocr_data: List[Dict], client=None, use_gemini: bool = True) -> str:
    """
    تجميع أسطر الـ OCR حسب رقم الصفحة، ثم إرسال كل صفحة مستقلة للـ LLM
    لتفادي الاقتطاع وضمان إخراج كافة الصفحات بنفس تنسيق النوت بوك.
    """
    pages_dict: Dict[int, List[str]] = {}
    for item in ocr_data:
        pages_dict.setdefault(item["page"], []).append(item["text"])

    if not pages_dict:
        return ""

    if use_gemini and client is None:
        client = _get_gemini_client()

    full_formatted_document = []
    logger.info(f"بدء تنسيق وتدقيق {len(pages_dict)} صفحات عبر LLM صفحة بصفحة...")

    for page_num in sorted(pages_dict.keys()):
        lines = pages_dict[page_num]
        raw_page_text = "\n".join(lines)

        logger.info(f"جاري تدقيق وتنسيق الصفحة {page_num} من {len(pages_dict)}...")

        prompt = f"""
أنت خبير تدقيق وتصحيح نصوص الـ OCR. أمامك النص الخام المستخرج من الصفحة رقم ({page_num}) من المستند.

المطلوب منك بدقة:
1. تصحيح الأخطاء الإملائية والمطبعية والكلمات المقطوعة الناتجة عن الـ OCR.
2. إعادة تنسيق النص بأسلوب مرتب ومقروء (عناوين، بنود، فقرات).
3. الحفاظ الكامل على جميع البيانات، الأسماء، الأرقام، والتواريخ دون اختصار أو حذف أي جزء.
4. ابدأ استجابتك مباشرة بـ "--- صفحة {page_num} ---" ثم اتبعها بالنص المنسق دون كتابة أي مقدمات أو شروحات.

النص الخام للصفحة {page_num}:
{raw_page_text}
""".strip()

        if use_gemini and client is not None:
            try:
                response = client.models.generate_content(
                    model=CONFIG["gemini_model"],
                    contents=prompt
                )
                page_output = (response.text or "").strip()
                full_formatted_document.append(page_output)
            except Exception as e:
                logger.warning(f"❌ حدث خطأ أثناء معالجة الصفحة {page_num} عبر Gemini: {e}")
                full_formatted_document.append(f"--- صفحة {page_num} ---\n{raw_page_text}")
        else:
            full_formatted_document.append(f"--- صفحة {page_num} ---\n{raw_page_text}")

    return "\n\n" + "\n\n".join(full_formatted_document)

def save_formatted_document(text: str, output_dir: str) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "formatted_document.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)

# ============================================================================
# Main Public API Pipeline
# ============================================================================
def extract(file_path: str, refine: bool = True, output_dir: str | None = None) -> dict:
    """
    المحرك الرئيسي لتشغيل مسار الـ OCR والتنسيق المطابق للنوت بوك.
    """
    path = Path(file_path)
    out_dir = output_dir or CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "status": "error",
        "metadata": {
            "file_name": path.name,
            "file_type": detect_file_type(file_path),
            "num_pages": 0,
            "method": None,
            "gemini_used": False,
            "output_dir": str(out_dir),
        },
        "raw_text": "",
        "formatted_text": "",
        "clean_text": "",
        "raw_ocr_data": [],
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"الملف غير موجود: {file_path}")
        return result

    try:
        pages = load_pages(str(path), dpi=int(CONFIG["dpi"]))
        result["metadata"]["num_pages"] = len(pages)

        # 1. الاستخراج المحلي
        raw_ocr_data = extract_text_local_ocr(pages)
        result["raw_ocr_data"] = raw_ocr_data

        if not raw_ocr_data:
            result["errors"].append("لم يتم استخراج أي نص عبر Tesseract.")
            return result

        # 2. حفظ المخرجات الخام والـ Immutable Copy
        raw_json_path, raw_txt_path = save_raw_ocr(raw_ocr_data, out_dir)
        RAW_OCR_IMMUTABLE = copy.deepcopy(raw_ocr_data)

        raw_pages: Dict[int, List[str]] = {}
        for item in raw_ocr_data:
            raw_pages.setdefault(item["page"], []).append(item["text"])
        
        raw_text = "\n\n".join("\n".join(raw_pages[p]) for p in sorted(raw_pages))
        result["raw_text"] = raw_text

        # 3. التنسيق والتدقيق عبر Gemini
        use_gemini = bool(refine)
        client = _get_gemini_client() if use_gemini else None

        formatted_document_text = clean_and_format_by_page(
            RAW_OCR_IMMUTABLE,
            client=client,
            use_gemini=use_gemini
        )

        formatted_path = save_formatted_document(formatted_document_text, out_dir)

        # 4. تعبئة المخرجات ومتغيرات الشات بوت للتربيط
        FINAL_CHATBOT_CONTEXT = formatted_document_text

        result["formatted_text"] = formatted_document_text
        result["clean_text"] = formatted_document_text
        result["FINAL_CHATBOT_CONTEXT"] = FINAL_CHATBOT_CONTEXT
        result["RAW_OCR_IMMUTABLE"] = RAW_OCR_IMMUTABLE

        result["metadata"]["method"] = (
            "tesseract_ocr+gemini_page_format"
            if use_gemini and client is not None
            else "tesseract_ocr"
        )
        result["metadata"]["gemini_used"] = bool(use_gemini and client is not None)
        result["metadata"]["raw_json_path"] = raw_json_path
        result["metadata"]["raw_txt_path"] = raw_txt_path
        result["metadata"]["formatted_output_path"] = formatted_path

        result["status"] = "ok"
        return result

    except Exception as exc:
        logger.exception("فشل مسار الـ OCR والتدقيق للملف %s", file_path)
        result["errors"].append(str(exc))
        return result
