# -*- coding: utf-8 -*-
"""
Stage 1 - OCR + Arabic document formatting.

This module mirrors the logic of contracts_ocr_pipeline.ipynb:

1. PDF/image -> 300 DPI images with PyMuPDF.
2. Local Tesseract OCR using ara+eng and --oem 3 --psm 6.
3. Preserve raw OCR line/page structure.
4. Send EACH PAGE independently to Gemini for correction and formatting.
5. Keep all names, numbers, dates and legal content; never summarize.
6. If Gemini fails, keep the raw OCR page instead of losing data.
7. Save the raw and formatted outputs.
8. Return the final formatted document for the chatbot/RAG stage.

Unlike the Colab notebook, this module has no google.colab dependency and accepts
a normal filesystem path.

Public API:
    extract(file_path, refine=True) -> dict
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

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

CONFIG = {
    "output_dir": os.environ.get("OCR_OUTPUT_DIR", "output"),
    "langs": ["ar", "eng"],
    "dpi": 300,
    "tesseract_config": r"--oem 3 --psm 6",
    "gemini_model": os.environ.get("GEMINI_OCR_MODEL", "gemini-3.6-flash"),
}

PAGE_SEPARATOR = "\n\n--- صفحة جديدة ---\n\n"


# ============================================================================
# Gemini client
# ============================================================================
def _get_gemini_api_key() -> str:
    """Read the Gemini key from the environment."""
    return os.environ.get("GEMINI_API_KEY", "").strip()


def gemini_available() -> bool:
    return HAS_GENAI and bool(_get_gemini_api_key())


def _get_gemini_client():
    if not gemini_available():
        return None
    return genai.Client(api_key=_get_gemini_api_key())


# Same page-level instruction used by the notebook, with the same
# preservation requirements. It intentionally does not ask the model to
# invent or summarize legal content.
def _build_page_prompt(page_num: int, raw_page_text: str) -> str:
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
    """Keep the notebook's page marker but remove accidental leading chatter."""
    text = (text or "").strip()
    if not text:
        return ""

    marker = f"--- صفحة {page_num} ---"
    if marker in text:
        text = text[text.find(marker):]

    # Remove common model lead-ins only when they appear before the page marker.
    text = re.sub(
        r"\A(?:بالتأكيد|بالطبع|إليك|فيما يلي|بعد التصحيح|النص المصحح)[^\n]*\n",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    return text


def format_page_with_gemini(client, page_num: int, raw_page_text: str) -> str:
    """Format one page independently, exactly as the notebook's core stage."""
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

        logger.warning(
            "Gemini returned an empty result for page %d; keeping raw OCR.",
            page_num,
        )
    except Exception as exc:
        logger.warning(
            "Gemini failed on page %d: %s; keeping raw OCR.",
            page_num,
            exc,
        )

    return f"--- صفحة {page_num} ---\n{raw_page_text}"


# ============================================================================
# File loading - mirrors notebook: PDF -> 300 DPI images
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
    """
    Convert the input into PIL RGB pages.

    For PDFs this intentionally rasterises every page at 300 DPI, matching
    the notebook instead of trying to use an existing PDF text layer.
    """
    if not HAS_FITZ:
        raise RuntimeError("PyMuPDF (fitz) غير مثبت.")

    ext = Path(input_path).suffix.lower()
    pages = []

    if ext == ".pdf":
        with fitz.open(input_path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(
                    io.BytesIO(pix.tobytes("png"))
                ).convert("RGB")
                pages.append(img)

    elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
        pages = [Image.open(input_path).convert("RGB")]

    else:
        raise ValueError(
            f"صيغة غير مدعومة في مسار OCR: {ext}. "
            "المسار الأساسي المطابق للـ Notebook يدعم PDF وJPG/JPEG/PNG."
        )

    if not pages:
        raise ValueError("لم يتم استخراج أي صفحة من الملف.")

    return pages


# ============================================================================
# Local Tesseract OCR - mirrors notebook exactly
# ============================================================================
def extract_text_local_ocr(pages) -> List[Dict]:
    """Extract OCR line-by-line with ara+eng and --oem 3 --psm 6."""
    if not HAS_TESSERACT:
        raise RuntimeError(
            "Tesseract غير متاح. ثبّت pytesseract وTesseract مع اللغة العربية، "
            "أو اضبط TESSERACT_CMD."
        )

    raw_ocr_data: List[Dict] = []
    custom_config = CONFIG["tesseract_config"]

    for page_idx, page_image in enumerate(pages):
        page_number = page_idx + 1
        logger.info(
            "جاري معالجة الصفحة %d من %d محلياً...",
            page_number,
            len(pages),
        )

        try:
            text = pytesseract.image_to_string(
                page_image,
                lang="ara+eng",
                config=custom_config,
            )

            lines = [
                line.strip()
                for line in text.split("\n")
                if line.strip()
            ]

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
            logger.warning(
                "خطأ في معالجة الصفحة %d: %s",
                page_number,
                exc,
            )

    return raw_ocr_data


# ============================================================================
# Raw OCR persistence - mirrors notebook
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
            for item in sorted(
                lines_by_page[page],
                key=lambda x: x["line_number"],
            ):
                f.write((item["text"] or "") + "\n")
            f.write("\n")

    return str(raw_json_path), str(raw_txt_path)


# ============================================================================
# Page-by-page Gemini formatting - mirrors notebook
# ============================================================================
def clean_and_format_by_page(
    ocr_data: List[Dict],
    client=None,
    use_gemini: bool = True,
) -> str:
    """
    Group OCR lines by page and send every page independently to Gemini.

    This is intentionally page-based rather than whole-document/chunk based:
    it prevents later pages from being dropped when a model response gets long.
    """
    pages_dict: Dict[int, List[str]] = {}

    for item in ocr_data:
        pages_dict.setdefault(item["page"], []).append(item["text"])

    if not pages_dict:
        return ""

    if use_gemini and client is None:
        client = _get_gemini_client()

    full_formatted_document: List[str] = []

    logger.info(
        "بدء تنسيق وتدقيق %d صفحات عبر LLM صفحة بصفحة...",
        len(pages_dict),
    )

    for page_num in sorted(pages_dict):
        lines = pages_dict[page_num]
        raw_page_text = "\n".join(lines)

        logger.info(
            "جاري تدقيق وتنسيق الصفحة %d من %d...",
            page_num,
            len(pages_dict),
        )

        if use_gemini and client is not None:
            page_output = format_page_with_gemini(
                client,
                page_num,
                raw_page_text,
            )
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
# Optional compatibility helpers for non-OCR text documents
# ============================================================================
def extract_word(docx_path: str) -> List[str]:
    if not HAS_DOCX:
        raise RuntimeError("python-docx غير مثبت — لا يمكن قراءة ملفات Word.")

    doc = docx.Document(docx_path)
    text = "\n".join(
        p.text for p in doc.paragraphs if p.text.strip()
    )
    return [text]


def extract_plain_text(file_path: str) -> List[str]:
    return [
        Path(file_path).read_text(
            encoding="utf-8",
            errors="ignore",
        )
    ]


# ============================================================================
# Public API
# ============================================================================
def extract(
    file_path: str,
    output_dir: str = "output",
    refine: bool = True,
) -> dict:
    """
    Entry point for the whole pipeline: PDF/image -> Tesseract OCR (ara+eng)
    -> per-page Gemini correction -> saved raw/formatted output.

    Parameters
    ----------
    file_path:
        Path to the input PDF or image (replaces Colab's files.upload()).
    output_dir:
        Directory for raw_ocr.json, raw_ocr.txt and formatted_document.txt.
        Defaults to "output".
    refine:
        If True, send each page to Gemini for spelling/formatting
        correction after local OCR. If Gemini is unavailable or a page
        fails, that page falls back to the raw OCR text (no data lost).

    Returns
    -------
    dict:
        status           "ok" | "error"
        metadata         file_name, file_type, num_pages, method,
                          gemini_used, output_dir, and saved file paths
        raw_text         raw OCR text, joined page by page
        formatted_text   final text after Gemini correction (or raw
                          text per page if Gemini was skipped/failed)
        clean_text       alias of formatted_text (kept for callers that
                          expect a "clean_text" key)
        clauses          reserved for a later clause-segmentation stage
        raw_ocr_data     full line-level OCR records (page, line_id,
                          line_number, text, confidence, bbox)
        errors           list of error messages, if any
    """
    path = Path(file_path)
    out_dir = output_dir or CONFIG["output_dir"]

    result = {
        "status": "error",
        "metadata": {
            "file_name": path.name,
            "file_type": detect_file_type(file_path),
            "num_pages": 0,
            "method": None,
            "num_clauses": 0,
            "gemini_used": False,
            "output_dir": str(out_dir),
        },
        "raw_text": "",
        "formatted_text": "",
        "clean_text": "",
        "clauses": [],
        "raw_ocr_data": [],
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"الملف غير موجود: {file_path}")
        return result

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        result["errors"].append(
            f"نوع ملف غير مدعوم: {path.suffix}"
        )
        return result

    try:
        # The notebook's OCR path is PDF/image based.
        if path.suffix.lower() in {
            ".pdf", ".png", ".jpg", ".jpeg",
            ".bmp", ".webp", ".tiff",
        }:
            pages = load_pages(
                str(path),
                dpi=int(CONFIG["dpi"]),
            )

            result["metadata"]["num_pages"] = len(pages)

            raw_ocr_data = extract_text_local_ocr(pages)
            result["raw_ocr_data"] = raw_ocr_data

            if not raw_ocr_data:
                result["errors"].append(
                    "لم يتم استخراج أي نص. تأكد من جودة المستند "
                    "وتثبيت Tesseract واللغة العربية."
                )
                return result

            raw_json_path, raw_txt_path = save_raw_ocr(
                raw_ocr_data,
                out_dir,
            )

            raw_pages: Dict[int, List[str]] = {}
            for item in raw_ocr_data:
                raw_pages.setdefault(item["page"], []).append(item["text"])

            raw_text = "\n\n".join(
                "\n".join(raw_pages[p])
                for p in sorted(raw_pages)
            )

            result["raw_text"] = raw_text

            use_gemini = bool(refine)
            client = _get_gemini_client() if use_gemini else None

            if use_gemini and client is None:
                logger.warning(
                    "GEMINI_API_KEY أو google-genai غير متاح؛ "
                    "سيتم الاحتفاظ بالنص الخام."
                )

            formatted_text = clean_and_format_by_page(
                raw_ocr_data,
                client=client,
                use_gemini=use_gemini,
            )

            formatted_path = save_formatted_document(
                formatted_text,
                out_dir,
            )

            result["formatted_text"] = formatted_text
            result["clean_text"] = formatted_text
            result["metadata"]["method"] = (
                "tesseract_ocr+gemini_page_format"
                if use_gemini and client is not None
                else "tesseract_ocr"
            )
            result["metadata"]["gemini_used"] = bool(
                use_gemini and client is not None
            )
            result["metadata"]["raw_json_path"] = raw_json_path
            result["metadata"]["raw_txt_path"] = raw_txt_path
            result["metadata"]["formatted_output_path"] = formatted_path

        elif path.suffix.lower() in {".docx", ".doc"}:
            pages = extract_word(str(path))
            result["metadata"]["num_pages"] = 1
            result["raw_text"] = pages[0]
            result["formatted_text"] = pages[0]
            result["clean_text"] = pages[0]
            result["metadata"]["method"] = "docx_direct"

        elif path.suffix.lower() in {".txt", ".csv", ".json"}:
            pages = extract_plain_text(str(path))
            result["metadata"]["num_pages"] = 1
            result["raw_text"] = pages[0]
            result["formatted_text"] = pages[0]
            result["clean_text"] = pages[0]
            result["metadata"]["method"] = "plain_text"

        else:
            result["errors"].append(
                f"نوع الملف غير مدعوم: {path.suffix}"
            )
            return result

        result["status"] = "ok"
        return result

    except Exception as exc:
        logger.exception("OCR pipeline failed for %s", file_path)
        result["errors"].append(str(exc))
        return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m modules.ocr <file>")
        raise SystemExit(1)

    result = extract(sys.argv[1])

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )[:8000]
    )
