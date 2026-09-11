import re

def _normalize_text(value):
    if value is None:
        return ""

    text = str(value).strip()

    replacements = {
        "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه",
        "ً": "", "ٌ": "", "ٍ": "", "َ": "", "ُ": "", "ِ": "", "ّ": "", "ْ": "", "ـ": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def _extract_all_numbers(text):
    if not text:
        return []
    cleaned = str(text).replace(",", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", cleaned)
    return [float(n) for n in numbers]

def _value_present_in_contract(value, contract_text):
    if value is None:
        return True

    value_str = str(value).strip()
    if not value_str:
        return True

    normalized_contract = _normalize_text(contract_text)
    normalized_value = _normalize_text(value_str)

    if normalized_value in normalized_contract:
        return True

    val_numbers = _extract_all_numbers(value_str)
    if val_numbers:
        contract_numbers = _extract_all_numbers(contract_text)
        if all(num in contract_numbers for num in val_numbers):
            return True

    stop_words = {"في", "من", "على", "عن", "إلى", "البدء", "تاريخ", "شروط", "اسم", "و", "أو"}
    words = [w for w in normalized_value.split() if len(w) > 2 and w not in stop_words]

    if words:
        matched_words_count = sum(1 for word in words if word in normalized_contract)
        match_ratio = matched_words_count / len(words)
        if match_ratio >= 0.6:
            return True

    return False

def _flatten_values(obj):
    values = []
    if isinstance(obj, dict):
        for value in obj.values():
            values.extend(_flatten_values(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(_flatten_values(value))
    elif obj is not None:
        value = str(obj).strip()
        if value:
            values.append(value)
    return values

def _check_required_data(fields, parties, contract_text):
    issues = []

    for key, value in fields.items():
        if key == "parties" or isinstance(value, (dict, list)) or value is None:
            continue
        if not _value_present_in_contract(value, contract_text):
            issues.append(f"Extracted value not represented in contract: {value}")

    party_values = _flatten_values(parties)
    for value in party_values:
        if value in ["صاحب العمل", "الموظف", "الطرف الأول", "الطرف الثاني", "البائع", "المشتري", "المؤجر", "المستأجر"]:
            continue
        if not _value_present_in_contract(value, contract_text):
            issues.append(f"Party data not represented in contract: {value}")

    return issues

def _check_placeholders(contract_text, missing_fields):
    issues = []
    raw_placeholders = re.findall(r"\[([^\]]+)\]", contract_text)

    allowed_placeholders = set()
    for field in missing_fields:
        field = str(field).strip()
        allowed_placeholders.add(_normalize_text(field))
        allowed_placeholders.add(_normalize_text(f"placeholder: {field}"))

    for placeholder in raw_placeholders:
        cleaned = placeholder.replace("*", "").strip()
        normalized = _normalize_text(cleaned)

        if normalized in ["placeholder", ""] or "يرجى" in normalized or "ادخال" in normalized or "شروط" in normalized or "أدخل" in normalized:
            if missing_fields:
                continue

        if normalized not in allowed_placeholders:
            issues.append(f"Unauthorized placeholder: [{placeholder}]")

    return issues

def _check_citations(contract_text, retrieved_articles):
    issues = []
    retrieved_numbers = set()

    for article in retrieved_articles:
        article_number = str(article.get("article_number", "")).strip()
        if article_number:
            retrieved_numbers.add(article_number)

    patterns = [
        r"(?:المادة|مادة)\s*[\(（]?\s*(\d+)\s*[\)）]?\s*(?:من\s+)?(?:قانون|القانون)",
        r"(?:وفقاً|وفقا|استناداً|استنادا|طبقاً|طبقا|بموجب)\s+ل?المادة\s*[\(（]?\s*(\d+)\s*[\)）]?",
        r"(?:المادة|مادة)\s*[\(（]?\s*(\d+)\s*[\)）]?\s*من\s+(?:نفس\s+)?القانون",
        r"\barticle\s+(\d+)\s+of\b",
    ]

    cited_numbers = set()
    for pattern in patterns:
        matches = re.findall(pattern, contract_text, flags=re.IGNORECASE)
        for number in matches:
            cited_numbers.add(str(number))

    for number in cited_numbers:
        if number not in retrieved_numbers:
            issues.append(f"Unauthorized legal article citation: المادة {number}")

    return issues

def _check_empty_contract(contract_text):
    issues = []
    if not contract_text:
        issues.append("Contract is empty.")
    elif len(contract_text.strip()) < 300:
        issues.append("Contract is suspiciously short.")
    return issues

def validate_contract(contract_text, fields, parties, missing_fields, retrieved_articles):
    issues = []
    issues.extend(_check_empty_contract(contract_text))
    issues.extend(_check_required_data(fields, parties, contract_text))
    issues.extend(_check_placeholders(contract_text, missing_fields))
    issues.extend(_check_citations(contract_text, retrieved_articles))

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "issue_count": len(issues)
    }