# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dotenv import load_dotenv

# تحميل مفتاح API تلقائياً من ملف .env المحلي
load_dotenv()

import io
import json
import logging
import re
import copy
from pathlib import Path
from typing import List, Dict, Tuple, Any

logger = logging.getLogger("contract_ai.ocr")

# ---------------------------------------------------------------------------
# المكتبات
# ---------------------------------------------------------------------------
try:
    import pymupdf as fitz
    HAS_FITZ = True
except ImportError:
    try:
        import fitz
        HAS_FITZ = True
    except ImportError:
        HAS_FITZ = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff",
    ".docx", ".doc",
    ".txt", ".csv", ".json",
}

# إعدادات الـ OCR والـ LLM
CONFIG: Dict[str, Any] = {
    "output_dir": os.environ.get("OCR_OUTPUT_DIR", "outputs"),
    "dpi": 300,
    "gemini_model": os.environ.get("GEMINI_OCR_MODEL", "gemini-2.5-flash"),
}


# ============================================================================
# Gemini Client & Prompt Helpers
# ============================================================================
def _get_gemini_api_key() -> str:
    """قراءة مفتاح API الخاص بـ Gemini من البيئة."""
    return os.environ.get("GEMINI_API_KEY", "").strip()


def gemini_available() -> bool:
    return HAS_GENAI and bool(_get_gemini_api_key())


def _get_gemini_client():
    if not gemini_available():
        return None
    return genai.Client(api_key=_get_gemini_api_key())


def _build_page_image_prompt(page_num: int) -> str:
    """الـ Prompt المخصص لاستخراج وتنسيق النص المباشر من صورة الصفحة."""
    return f"""
أنت خبير قانوني ومختص في تحويل وصياغة العقود القانونية من الصور والمستندات بدقة متناهية.
أمامك صورة للصفحة رقم ({page_num}) من عقد قانوني.

المطلوب منك بدقة عالية:
1. استخراج وقراءة كافة النصوص الموجودة في صورة الصفحة بالكامل دون حذف أو اختصار.
2. تصحيح أي كلمات غير واضحة وإعادة تنسيق النص بأسلوب قانوني مرتب (عناوين، بنود، فقرات، أرقام).
3. الحفاظ الكامل والدقيق على جميع الأسماء، المبالغ، التواريخ، والبنود دون أي تعديل في المعنى.
4. عدم إضافة أي معلومات أو استنتاجات من خارج الصورة.
5. ابدأ استجابتك مباشرة بـ "--- صفحة {page_num} ---" ثم اتبعها بالنص المنسق دون أي مقدمات أو كلام إضافي.
""".strip()


def _strip_unwanted_preamble(text: str, page_num: int) -> str:
    """إزالة أي مقدمات تواصلية يضيفها النموذج."""
    text = (text or "").strip()
    if not text:
        return ""

    marker = f"--- صفحة {page_num} ---"
    if marker in text:
        text = text[text.find(marker):]

    text = re.sub(
        r"\A(?:بالتأكيد|بالطبع|إليك|فيما يلي|بعد التصحيح|النص المصحح)[^\n]*\n",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    return text


def process_page_image_with_gemini(client, page_num: int, image: Image.Image) -> str:
    """إرسال صورة الصفحة مباشرة إلى Gemini Vision واستخراج النص المنسق."""
    prompt = _build_page_image_prompt(page_num)

    try:
        response = client.models.generate_content(
            model=CONFIG["gemini_model"],
            contents=[image, prompt],
        )
        result = _strip_unwanted_preamble(
            getattr(response, "text", "") or "",
            page_num,
        )
        if result:
            return result

        logger.warning("أرجع Gemini نتيجة فارغة للصفحة %d.", page_num)
    except Exception as exc:
        logger.error("فشلت معالجة Gemini Vision للصفحة %d: %s", page_num, exc)
        raise exc

    return f"--- صفحة {page_num} ---\n[تعذر استخراج النص من هذه الصفحة]"


# ============================================================================
# Page Loading (PDF -> PIL Images)
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


def load_pages(input_path: str, dpi: int = 300) -> List[Image.Image]:
    """تحويل صفحات الـ PDF إلى صور PIL بوضوح high resolution."""
    if not HAS_FITZ:
        raise RuntimeError("مكتبة PyMuPDF غير مثبتة. قم بتثبيتها عبر pip install pymupdf")

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
        raise ValueError("لم يتم استخراج أي صفحات من الملف.")

    return pages


# ============================================================================
# Saving Helpers
# ============================================================================
def save_formatted_document(text: str, output_dir: str) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    path = output / "formatted_document.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def extract_word(docx_path: str) -> List[str]:
    if not HAS_DOCX:
        raise RuntimeError("مكتبة python-docx غير مثبتة.")

    doc = docx.Document(docx_path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [text]


def extract_plain_text(file_path: str) -> List[str]:
    return [Path(file_path).read_text(encoding="utf-8", errors="ignore")]


# ============================================================================
# Main Entrypoint
# ============================================================================
def extract(
    file_path: str,
    refine: bool = True,
    output_dir: str | None = None,
) -> dict:
    """
    الدالة الرئيسية للاستخراج (تعتمد المسميات والتربيطات الخاصة بالمشروع)
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

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        result["errors"].append(f"نوع ملف غير مدعوم: {path.suffix}")
        return result

    try:
        if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}:
            client = _get_gemini_client()
            if not client:
                result["errors"].append("تنسيق Gemini غير متاح! تأكد من وجود GEMINI_API_KEY في ملف .env")
                return result

            pages = load_pages(str(path), dpi=int(CONFIG["dpi"]))
            result["metadata"]["num_pages"] = len(pages)

            formatted_pages = []
            raw_ocr_data = []

            # استخراج مباشر عبر Gemini Vision لكل صفحة
            for idx, img in enumerate(pages):
                page_num = idx + 1
                logger.info("جاري تحليل واستخراج النص للصفحة %d عبر Gemini Vision...", page_num)
                
                page_text = process_page_image_with_gemini(client, page_num, img)
                formatted_pages.append(page_text)

                # محاكاة السجل لتجهيز RAW_OCR_IMMUTABLE
                raw_ocr_data.append({
                    "page": page_num,
                    "line_id": f"p{page_num}_all",
                    "line_number": 1,
                    "text": page_text,
                    "confidence": 1.0,
                    "bbox": None,
                })

            full_formatted_text = "\n\n".join(formatted_pages)
            formatted_path = save_formatted_document(full_formatted_text, out_dir)

            # التربيطات الخاصة بالنظام
            result["raw_text"] = full_formatted_text
            result["formatted_text"] = full_formatted_text
            result["clean_text"] = full_formatted_text
            result["FINAL_CHATBOT_CONTEXT"] = full_formatted_text
            result["RAW_OCR_IMMUTABLE"] = raw_ocr_data
            result["raw_ocr_data"] = raw_ocr_data

            result["metadata"]["method"] = "gemini_vision_direct_ocr"
            result["metadata"]["gemini_used"] = True
            result["metadata"]["formatted_output_path"] = formatted_path

        elif path.suffix.lower() in {".docx", ".doc"}:
            pages = extract_word(str(path))
            result["metadata"]["num_pages"] = 1
            result["raw_text"] = pages[0]
            result["formatted_text"] = pages[0]
            result["clean_text"] = pages[0]
            result["FINAL_CHATBOT_CONTEXT"] = pages[0]
            result["metadata"]["method"] = "docx_direct"

        elif path.suffix.lower() in {".txt", ".csv", ".json"}:
            pages = extract_plain_text(str(path))
            result["metadata"]["num_pages"] = 1
            result["raw_text"] = pages[0]
            result["formatted_text"] = pages[0]
            result["clean_text"] = pages[0]
            result["FINAL_CHATBOT_CONTEXT"] = pages[0]
            result["metadata"]["method"] = "plain_text"

        result["status"] = "ok"
        return result

    except Exception as exc:
        logger.exception("فشل في معالجة OCR للملف %s", file_path)
        result["errors"].append(str(exc))
        return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.ocr <file_path>")
        sys.exit(1)

    res = extract(sys.argv[1])
    print(json.dumps(res, ensure_ascii=False, indent=2)[:4000])
