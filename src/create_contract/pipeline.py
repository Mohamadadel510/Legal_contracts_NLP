import gzip
import json
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from src.create_contract.llm_client import generate
from src.create_contract.validator import validate_contract

# ============================================================
# CONTRACT TYPES & LOCAL RAG PATH
# ============================================================

CONTRACT_TYPES = [
    "agricultural",
    "commercial",
    "company",
    "consumer",
    "employment",
    "general",
    "residential",
    "sale",
    "service",
    "supply",
]

POINTS_PATH = Path("data/legallens_legal_only_points.json.gz")

print("RAG path:", POINTS_PATH)
print("Exists:", POINTS_PATH.exists())

# ============================================================
# EMBEDDING & RAG LOADING
# ============================================================

MODEL_NAME = "BAAI/bge-m3"

_model = None
_vectors = None
_metadata = None
_initialized = False

def _load_model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def _load_points():
    global _vectors, _metadata, _initialized
    if _initialized:
        return

    if not POINTS_PATH.exists():
        raise FileNotFoundError(f"RAG index not found at:\n{POINTS_PATH.resolve()}")

    print("Loading LegalLens RAG index (compressed)...")
    
    # قراءة الملف المضغوط بصيغة .gz باستخدام مكتبة gzip
    with gzip.open(POINTS_PATH, "rt", encoding="utf-8") as f:
        points = json.load(f)

    if not points:
        raise ValueError("The RAG export contains no points.")

    vectors = []
    metadata = []

    for point in points:
        vector = point.get("vector")
        payload = point.get("payload", {})
        if vector is None:
            continue

        contract_types = payload.get("contract_types", [])
        if isinstance(contract_types, str):
            contract_types = [contract_types]

        metadata.append({
            "id": point.get("id"),
            "law_id": payload.get("law_id", ""),
            "law_name": payload.get("law_name", ""),
            "law_number": payload.get("law_number", ""),
            "article_number": str(payload.get("article_number", payload.get("article", ""))),
            "article_text": payload.get("text", payload.get("article_text", "")),
            "source": payload.get("source", ""),
            "contract_types": contract_types,
        })
        vectors.append(vector)

    if vectors:
        _vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _vectors = _vectors / norms

    _metadata = metadata
    _initialized = True
    print(f"Loaded {_vectors.shape[0]} legal vectors successfully!")

def initialize():
    _load_model()
    _load_points()

def _embed_query(query):
    model = _load_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embedding.astype(np.float32)

def retrieve(clause_text: str, contract_type: str, top_k: int = 3):
    if not isinstance(clause_text, str) or not clause_text.strip():
        return []
    if contract_type not in CONTRACT_TYPES:
        contract_type = "general"

    initialize()
    if _vectors is None or len(_metadata) == 0:
        return []

    query_vector = _embed_query(clause_text)

    if contract_type == "general":
        allowed_indices = list(range(len(_metadata)))
    else:
        allowed_indices = [
            idx for idx, item in enumerate(_metadata)
            if contract_type in item.get("contract_types", [])
        ]
        if not allowed_indices:
            allowed_indices = [
                idx for idx, item in enumerate(_metadata)
                if "general" in item.get("contract_types", [])
            ]

    if not allowed_indices:
        return []

    candidate_vectors = _vectors[allowed_indices]
    scores = candidate_vectors @ query_vector[0]

    k = min(top_k, len(scores))
    if k == 0:
        return []

    candidate_positions = np.argpartition(-scores, k - 1)[:k]
    candidate_positions = candidate_positions[np.argsort(-scores[candidate_positions])]

    results = []
    for rank, position in enumerate(candidate_positions, start=1):
        original_idx = allowed_indices[position]
        item = _metadata[original_idx]
        results.append({
            "rank": rank,
            "law_id": item["law_id"],
            "law_name": item["law_name"],
            "law_number": item["law_number"],
            "article_number": item["article_number"],
            "article_text": item["article_text"],
            "score": round(float(scores[position]), 6),
            "contract_types": item["contract_types"],
        })
    return results

def build_legal_context(retrieved_articles):
    if not retrieved_articles:
        return "لا توجد مواد قانونية مسترجعة من قاعدة المعرفة."
    
    context_lines = []
    for i, article in enumerate(retrieved_articles, start=1):
        context_lines.append(f"""
[{i}]
القانون: {article['law_name']}
رقم القانون: {article['law_number']}
المادة: {article['article_number']}
درجة الصلة: {article['score']}

نص المادة:
{article['article_text']}
""".strip())
    return "\n\n".join(context_lines)

# ============================================================
# STAGE 1: CLASSIFICATION
# ============================================================

KEYWORD_RULES = {
    "residential": ["إيجار شقة", "ايجار شقة", "سكني", "منزل", "بيت"],
    "commercial": ["إيجار تجاري", "ايجار تجاري", "محل تجاري"],
    "agricultural": ["أرض زراعية", "ارض زراعية", "زراعي", "محصول"],
    "employment": ["موظف", "وظيفة", "عقد عمل", "راتب", "مرتب", "عامل"],
    "sale": ["عقد بيع", "عقد شراء", "بيع شقة", "شراء شقة", "البائع", "المشتري"],
    "supply": ["توريد", "عقد توريد", "مورد"],
    "service": ["تقديم خدمة", "مقدم الخدمة", "عقد خدمة"],
    "company": ["شراكة", "شريك", "تأسيس شركة", "عقد شركة"],
    "consumer": ["ضمان المنتج", "حقوق المستهلك", "مستهلك"],
}

def suggest_contract_type(user_message):
    text = user_message.lower()
    scores = {ct: 0 for ct in CONTRACT_TYPES}
    for ct, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw.lower() in text:
                scores[ct] += 1
    best_type = max(scores, key=scores.get)
    if scores[best_type] > 0:
        return best_type
    return "general"

def confirm_contract_type(user_override_choice):
    if user_override_choice in CONTRACT_TYPES:
        return user_override_choice
    return "general"

# ============================================================
# STAGE 2: EXTRACTION
# ============================================================

EXTRACTION_SYSTEM_PROMPT = """
You are the Requirement Extraction component of LegalLens, an Egyptian legal-contract platform.
Extract ALL information explicitly provided by the user into precise, short, atomic fields in Arabic.
Return EXACTLY this JSON structure:
{
  "contract_type": "sale",
  "confidence": "high",
  "fields": {},
  "parties": {
    "party_a": {},
    "party_b": {}
  },
  "missing_fields": []
}
CRITICAL LANGUAGE RULE:
- Extract all fields, descriptions, and terms strictly in Arabic (Modern Standard Arabic).
- Do not mix English with Arabic.
"""

def _clean_json_output(raw_text):
    text = raw_text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return text[start:end + 1]

def extract_requirements(user_message, forced_contract_type=None):
    active_type = forced_contract_type or "general"
    user_prompt = f"""User Request:\n\"\"\"\n{user_message}\n\"\"\"\nContract Type: {active_type}\nReturn the required JSON object."""
    
    raw_text = generate(user_prompt=user_prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT, temperature=0.0)
    cleaned = _clean_json_output(raw_text)
    
    try:
        result = json.loads(cleaned)
    except Exception:
        fixed_str = re.sub(r'"\s*\n\s*"', '",\n"', cleaned)
        result = json.loads(fixed_str)
        
    result.setdefault("parties", {"party_a": {}, "party_b": {}})
    if forced_contract_type:
        result["contract_type"] = forced_contract_type
    return result

# ============================================================
# STAGE 3 & 4: DRAFTING
# ============================================================

DRAFTER_SYSTEM_PROMPT = """
أنت LegalLens، مساعد قانوني متخصص في صياغة العقود وفقاً للقانون المصري.
اكتب مسودة عقد واضحة ومنظمة باللغة العربية الفصحى بناءً على البيانات المستخرجة والمواد القانونية في RAG CONTEXT.
لا تخترع أي معلومات شخصية أو أرقام مواد غير موجودة في RAG.
أخرج مسودة العقد فقط دون شرح.
"""

def draft_contract(user_query, contract_type, fields, parties, missing_fields=None, top_k=5):
    if missing_fields is None:
        missing_fields = []

    retrieved_articles = retrieve(user_query, contract_type, top_k=top_k)
    legal_context = build_legal_context(retrieved_articles)

    drafting_request = f"""
نوع العقد: {contract_type}
بيانات المستخدم: {user_query}
البيانات المستخرجة: {json.dumps(fields, ensure_ascii=False, indent=2)}
الأطراف: {json.dumps(parties, ensure_ascii=False, indent=2)}
البيانات الناقصة: {json.dumps(missing_fields, ensure_ascii=False, indent=2)}

RAG CONTEXT:
{{RAG_CONTEXT}}
"""
    drafting_request = drafting_request.replace("{RAG_CONTEXT}", legal_context)

    contract_text = generate(user_prompt=drafting_request, system_prompt=DRAFTER_SYSTEM_PROMPT, temperature=0.1)

    return {
        "contract_text": contract_text,
        "retrieved_articles": retrieved_articles,
    }

# ============================================================
# ORCHESTRATOR / PIPELINE MAIN
# ============================================================

def create_contract(user_message, user_selected_contract_type=None, top_k=5, verbose=True):
    suggested_type = suggest_contract_type(user_message)
    contract_type = confirm_contract_type(user_selected_contract_type) if user_selected_contract_type else suggested_type

    stage1 = extract_requirements(user_message=user_message, forced_contract_type=contract_type)
    fields = stage1.get("fields", {})
    parties = stage1.get("parties", {"party_a": {}, "party_b": {}})
    missing_fields = stage1.get("missing_fields", [])

    draft_result = draft_contract(
        contract_type=contract_type,
        fields=fields,
        parties=parties,
        missing_fields=missing_fields,
        user_query=user_message,
        top_k=top_k,
    )

    validation = validate_contract(
        contract_text=draft_result["contract_text"],
        fields=fields,
        parties=parties,
        missing_fields=missing_fields,
        retrieved_articles=draft_result["retrieved_articles"]
    )

    return {
        "structured_requirements": stage1,
        "contract_type": contract_type,
        "contract_text": draft_result["contract_text"],
        "retrieved_articles": draft_result["retrieved_articles"],
        "validation": validation,
    }