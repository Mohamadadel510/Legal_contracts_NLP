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
# المكتبات الاختيارية ومسارات الويندوز
# ---------------------------------------------------------------------------
try:
    import pymupdf as fitz  # استخدام pymupdf بدلاً من الاستدعاء القديم لتفادي Deprecation Warning
    HAS_FITZ = True
except ImportError:
    try:
        import fitz
        HAS_FITZ = True
    except ImportError:
        HAS_FITZ = False

try:
    import pytesseract
    from PIL import Image
    
    # تحديد مسار Tesseract الشهير على نظام الويندوز حل مشكلة (لم يتم استخراج أي نص محلياً)
    tess_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tess_path):
        pytesseract.pytesseract.tesseract_cmd = tess_path
    elif os.environ.get("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD")
        
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

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

# إعدادات الـ OCR والـ LLM المطابقة للنوتبوك
CONFIG: Dict[str, Any] = {
    "output_dir": os.environ.get("OCR_OUTPUT_DIR", "outputs"),
    "langs": ["ar", "eng"],
    "dpi": 300,
    "chunk_max_lines": 40,
    "chunk_overlap_lines": 3,
    "tesseract_config": r"--oem 3 --psm 6",
    "gemini_model": os.environ.get("GEMINI_OCR_MODEL", "gemini-2.5-flash"),
}

PAGE_SEPARATOR = "\n\n--- صفحة جديدة ---\n\n"


# ============================================================================
# Gemini Client & Prompt Helpers (Matching Notebook)
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


def _build_page_prompt(page_num: int, raw_page_text: str) -> str:
    """الـ Prompt المعتمد بداخل النوتبوك لمعالجة وتنسيق النصوص صفحة بصفحة."""
    return f"""
أنت خبير تدقيق وتصحيح نصوص الـ OCR. أمامك النص الخام المستخرج من الصفحة رقم ({page_num}) من المستند.

المطلوب منك بدقة:
1. تصحيح الأخطاء الإملائية والمطبعية والكلمات المقطوعة الناتجة عن الـ OCR.
2. إعادة تنسيق النص بأسلوب مرتب ومقروء (عناوين، بنود، فقرات).
3. الحفاظ الكامل على جميع البيانات، الأسماء، الأرقام، والتواريخ دون اختصار أو حذف أي جزء.
4. لا تخترع أي معلومة غير موجودة في النص الخام.
5. لا تلخص النص ولا تعيد صياغته بما يغير معناه القانوني.
6. إذا كان جزء ما غير واضح ولا يمكن تصحيحه بثقة، اتركه كما هو بدل التخمين.
7. ابدأ استجابتك مباشرة بـ "--- صفحة {page_num} ---" ثم اتبعها بالنص المنسق دون كتابة أي مقدمات أو شروحات.

النص الخام للصفحة {page_num}:
{raw_page_text}
""".strip()


def _strip_unwanted_preamble(text: str, page_num: int) -> str:
    """إزالة أي مقدمات تواصلية يضيفها النموذج وإجباره على البدء باسم الصفحة."""
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


def format_page_with_gemini(client, page_num: int, raw_page_text: str) -> str:
    """معالجة وتدقيق الصفحة بشكل منفصل عبر Gemini."""
    prompt = _build_page_prompt(page_num, raw_page_text)

    try:
        response = client.models.generate_content(
            model=CONFIG["gemini_model"],
            contents=prompt,
        )
        result = _strip_unwanted_preamble(
            getattr(response, "text", "") or "",
            page_num,
        )
        if result:
            return result

        logger.warning("أرجع Gemini نتيجة فارغة للصفحة %d؛ سيتم الاحتفاظ بالنص الخام.", page_num)
    except Exception as exc:
        logger.warning("فشلت معالجة Gemini للصفحة %d: %s؛ سيتم استخدام النص الخام.", page_num, exc)

    return f"--- صفحة {page_num} ---\n{raw_page_text}"


# ============================================================================
# Page Loading (PDF -> Images at 300 DPI)
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
    """تحويل الـ PDF لصور بجودة 300DPI."""
    if not HAS_FITZ:
        raise RuntimeError("مكتبة PyMuPDF غير مثبتة.")

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
# Local Tesseract OCR
# ============================================================================
def extract_text_local_ocr(pages) -> List[Dict]:
    """استخراج أسطر النصوص محلياً عبر Tesseract."""
    if not HAS_TESSERACT:
        raise RuntimeError("تطبيق Tesseract أو مكتبة pytesseract غير متاحة.")

    raw_ocr_data: List[Dict] = []
    custom_config = CONFIG["tesseract_config"]

    for page_idx, page_image in enumerate(pages):
        page_number = page_idx + 1
        logger.info("جاري المعالجة المحلية للصفحة %d عبر Tesseract...", page_number)

        try:
            # استخراج النص مع دعم اللغة العربية والإنجليزي
            text = pytesseract.image_to_string(
                page_image,
                lang="ara+eng",
                config=custom_config,
            )

            lines = [line.strip() for line in text.split("\n") if line.strip()]

            for line_idx, line_text in enumerate(lines):
                raw_ocr_data.append({
                    "page": page_number,
                    "line_id": f"p{page_number}_l{line_idx + 1}",
                    "line_number": line_idx + 1,
                    "text": line_text,
                    "confidence": 1.0,
                    "bbox": None,
                })

        except Exception as exc:
            logger.warning("خطأ في معالجة الصفحة %d: %s", page_number, exc)

    return raw_ocr_data


# ============================================================================
# Saving & File Persistence
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
        for page in sorted(lines_by_page):
            f.write(f"===== صفحة {page} =====\n")
            for item in sorted(lines_by_page[page], key=lambda x: x["line_number"]):
                f.write((item["text"] or "") + "\n")
            f.write("\n")

    return str(raw_json_path), str(raw_txt_path)


# ============================================================================
# Page-by-Page Gemini Formatting Engine
# ============================================================================
def clean_and_format_by_page(
    ocr_data: List[Dict],
    client=None,
    use_gemini: bool = True,
) -> str:
    """تجميع أسطر OCR حسب الصفحات وإرسال كل صفحة بملفها المستقل لـ Gemini."""
    pages_dict: Dict[int, List[str]] = {}

    for item in ocr_data:
        pages_dict.setdefault(item["page"], []).append(item["text"])

    if not pages_dict:
        return ""

    if use_gemini and client is None:
        client = _get_gemini_client()

    full_formatted_document: List[str] = []

    for page_num in sorted(pages_dict):
        lines = pages_dict[page_num]
        raw_page_text = "\n".join(lines)

        if use_gemini and client is not None:
            page_output = format_page_with_gemini(client, page_num, raw_page_text)
        else:
            page_output = f"--- صفحة {page_num} ---\n{raw_page_text}"

        full_formatted_document.append(page_output)

    return "\n\n" + "\n\n".join(full_formatted_document)


def save_formatted_document(text: str, output_dir: str) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    path = output / "formatted_document.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


# ============================================================================
# Secondary Extractors
# ============================================================================
def extract_word(docx_path: str) -> List[str]:
    if not HAS_DOCX:
        raise RuntimeError("مكتبة python-docx غير مثبتة.")

    doc = docx.Document(docx_path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [text]


def extract_plain_text(file_path: str) -> List[str]:
    return [Path(file_path).read_text(encoding="utf-8", errors="ignore")]


# ============================================================================
# Main Public Entrypoint (المحافظة الكاملة على المفاتيح والتربيطات)
# ============================================================================
def extract(
    file_path: str,
    refine: bool = True,
    output_dir: str | None = None,
) -> dict:
    """
    الدالة الرئيسية المستدعاة في مشروعك من orchestrator.py
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
            pages = load_pages(str(path), dpi=int(CONFIG["dpi"]))
            result["metadata"]["num_pages"] = len(pages)

            # 1. الاستخراج المحلي عبر Tesseract
            raw_ocr_data = extract_text_local_ocr(pages)
            result["raw_ocr_data"] = raw_ocr_data

            if not raw_ocr_data:
                result["errors"].append("لم يتم استخراج أي نص محلياً عبر Tesseract. تأكد من تثبيت حزمة اللغة العربية Tesseract-OCR.")
                return result

            # 2. حفظ النص الخام وإنشاء نسخة RAW_OCR_IMMUTABLE
            raw_json_path, raw_txt_path = save_raw_ocr(raw_ocr_data, out_dir)
            RAW_OCR_IMMUTABLE = copy.deepcopy(raw_ocr_data)

            raw_pages: Dict[int, List[str]] = {}
            for item in raw_ocr_data:
                raw_pages.setdefault(item["page"], []).append(item["text"])

            raw_text = "\n\n".join("\n".join(raw_pages[p]) for p in sorted(raw_pages))
            result["raw_text"] = raw_text

            # 3. إمرار النصوص على Gemini LLM صفحة بصفحة
            use_gemini = bool(refine)
            client = _get_gemini_client() if use_gemini else None

            if use_gemini and client is None:
                logger.warning("GEMINI_API_KEY غير متاح بملف .env؛ سيتم استخدام النص الخام.")

            formatted_text = clean_and_format_by_page(
                RAW_OCR_IMMUTABLE,
                client=client,
                use_gemini=use_gemini,
            )

            formatted_path = save_formatted_document(formatted_text, out_dir)

            # 4. حفظ وتوفير جميع التربيطات البرمجية المطلوبة لباقي المشروع
            FINAL_CHATBOT_CONTEXT = formatted_text

            result["formatted_text"] = formatted_text
            result["clean_text"] = formatted_text
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
