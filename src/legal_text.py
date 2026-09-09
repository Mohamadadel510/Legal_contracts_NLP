# -*- coding: utf-8 -*-
"""
Repair for the legal corpus text.

The indexed articles were themselves produced by OCR, and 58% of them carry
visible damage: line-broken numbering, sentence-ending periods stranded on
their own lines, and a systematic ك -> ا substitution that turns كان into اان
and الدائن into الداين. Left alone, that text is what the app shows the user
as the legal basis for calling their clause void - and what the model is asked
to reason over.

Scope, deliberately narrow:

  * Layout is repaired unconditionally. Rejoining wrapped lines and folding
    ")1(" back into "(1)" cannot change a word.
  * Letters are repaired only from an explicit list of forms that are not
    Arabic words at all. No general "fix the spelling" pass: guessing at a
    statute's wording is worse than showing it slightly mangled.

This runs on read, not on the stored data. The index is unchanged, so
retrieval ranking still sees the damaged text - fixing that needs the corpus
rebuilt from a better source, which is a job for the RAG owner.
"""
from __future__ import annotations

import re

# Forms below are not valid Arabic words, so replacing them cannot destroy a
# meaning that was there. Frequencies are from data/legal_articles.json.
OCR_WORD_FIXES = {
    "اان": "كان",           # 573
    "اانت": "كانت",         # 139
    "واان": "وكان",         # 41
    "اافيا": "كافيا",       # 16
    "اافية": "كافية",
    "بضايع": "بضائع",
    "الموجر": "المؤجر",
    "الموجرة": "المؤجرة",
    "موجر": "مؤجر",
    "المستاجر": "المستأجر",
    "مستاجر": "مستأجر",
    "استاجر": "استأجر",
    "اذلك": "كذلك",
    "الداين": "الدائن",     # 235
    "الداينين": "الدائنين",  # 149
    "للداين": "للدائن",     # 70
    "الداينون": "الدائنون",  # 29
    "للداينين": "للدائنين",  # 22
    "داين": "دائن",         # 26
    "الشرااء": "الشركاء",   # 82
    "الحايز": "الحائز",     # 59
    "حايز": "حائز",         # 16
    "رييس": "رئيس",         # 53
    "الواالة": "الوكالة",   # 39
    "شييا": "شيئا",         # 30
    "نهاييا": "نهائيا",     # 29
    "جزييا": "جزئيا",       # 15
    "االعالن": "الإعلان",   # 19
    "االحوال": "الأحوال",   # 15
    "الوسايل": "الوسائل",
    "وسايل": "وسائل",
    "الموسسات": "المؤسسات",
    "موسسة": "مؤسسة",
    "قايمة": "قائمة",
    "الفوايد": "الفوائد",
    "سيي": "سيئ",
    "سيية": "سيئة",
}

# Damage this pass deliberately does not touch, so the limitation is visible
# to whoever reads the output rather than buried:
#
#   باالعالن / االضرار / اال خالل - a word-initial "اال" can expand to either
#   الإ or الأ, and the two are different words (الإضرار = causing harm,
#   الأضرار = damages). Guessing would silently rewrite a statute, so these
#   are left as they are. Fixing them properly means rebuilding the corpus
#   from a clean source - see docs/retriever.md.

# Stems safe to replace anywhere inside a word, because no Arabic word
# contains them: the word list above only catches the bare forms, and the
# corpus is full of prefixed and suffixed variants (للمستاجر, المستاجرين,
# والموجر) that would otherwise stay broken.
STEM_FIXES = {
    "مستاجر": "مستأجر",
    "موجر": "مؤجر",
    "حايز": "حائز",
    "شرااء": "شركاء",
    "وسايل": "وسائل",
    "موسس": "مؤسس",
    "فوايد": "فوائد",
    "بضايع": "بضائع",
    "قايم": "قائم",
    "رييس": "رئيس",
    "مسيول": "مسؤول",
}
_STEMS = re.compile("|".join(map(re.escape, STEM_FIXES)))

# "-يية" is not a possible Arabic ending; the carrier hamza was flattened.
# Covers القضايية, الابتدايية, الجنايية and the rest of the family at once.
_HAMZA_YA_ENDING = re.compile(r"يية\b")

# The OCR sometimes breaks after the definite article ("ال تحسينات"). A bare
# "ال" is not a word in Arabic, so rejoining it is safe.
_ORPHAN_AL = re.compile(r"(?<![ء-ي])ال\s+(?=[ء-ي])")

_NUMBERED_ITEM = re.compile(r"[–—\-]?\s*\)\s*(\d+)\s*\(\s*")
_ORPHAN_PERIOD = re.compile(r"\s*\n\s*\.\s*")
_HARD_WRAP = re.compile(r"\s*\n\s*")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def clean_article_text(text: str) -> str:
    """Make one indexed article readable. Returns "" for empty input."""
    if not text:
        return ""

    out = text.replace("\r", "")
    out = _NUMBERED_ITEM.sub(r" (\1) ", out)   # ")1(" spread over lines
    out = _ORPHAN_PERIOD.sub(". ", out)        # a period alone on its line
    out = _HARD_WRAP.sub(" ", out)             # unwrap mid-sentence breaks

    out = _ORPHAN_AL.sub("ال", out)
    out = _STEMS.sub(lambda m: STEM_FIXES[m.group(0)], out)
    out = _HAMZA_YA_ENDING.sub("ئية", out)
    out = re.sub(
        r"\b(" + "|".join(map(re.escape, OCR_WORD_FIXES)) + r")\b",
        lambda m: OCR_WORD_FIXES[m.group(1)],
        out,
    )

    out = _MULTISPACE.sub(" ", out)
    return out.strip()


# The Egyptian civil code ends at article 1149, and the other indexed laws are
# shorter still. 63 of the 1352 records carry a number above that - 4375, 2565,
# 1819 - which are page numbers or stray digits the corpus builder read as
# article numbers. Printing "المادة (4375)" under a clause is worse than
# printing nothing: it is a citation a lawyer can see is invented. The text
# itself is still real law, so keep it and drop only the number.
MAX_PLAUSIBLE_ARTICLE_NUMBER = 1200


def plausible_article_number(value) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if not text.isdigit():
        return True  # labels like "مكرر" are fine as they are
    return 1 <= int(text) <= MAX_PLAUSIBLE_ARTICLE_NUMBER


def clean_articles(articles: list[dict]) -> list[dict]:
    """Copy of `articles` with text repaired and impossible numbers dropped."""
    out = []
    for a in articles:
        item = {**a, "text": clean_article_text(a.get("text", ""))}
        if not plausible_article_number(item.get("article_number")):
            item["article_number_suspect"] = item.get("article_number")
            item["article_number"] = None
        out.append(item)
    return out


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads(
        (Path(__file__).resolve().parent.parent / "data" / "legal_articles.json")
        .read_text(encoding="utf-8")
    )
    for article in data[:3]:
        print("=== BEFORE ===")
        print(repr(article["text"][:200]))
        print("=== AFTER ===")
        print(clean_article_text(article["text"])[:200])
        print()
