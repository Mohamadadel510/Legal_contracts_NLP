from __future__ import annotations
import json, pickle, re
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"
ARTICLES_PATH = DATA_DIR / "legal_articles.json"
BM25_PATH = DATA_DIR / "bm25.pkl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
QDRANT_PATH = INDEX_DIR / "legal_rag_qdrant"
COLLECTION_NAME = "egyptian_legal_articles_contract_types"
MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

SUPPORTED_CONTRACT_TYPES = {
    "residential","agricultural","commercial","employment","company",
    "sale","supply","service","consumer","general"
}

QUERY_SYNONYMS = {
    "ايجار":"إيجار", "اجرة":"أجرة", "فلوس":"مقابل مالي",
    "مرتب":"أجر", "شغل":"عمل", "موظف":"عامل"
}

def normalize_arabic(text: str) -> str:
    text = str(text or "").replace("ـ","")
    text = re.sub(r"[إأآٱ]", "ا", text).replace("ى","ي")
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]","",text)
    return re.sub(r"\s+"," ",text).strip()

def tokenize_arabic(text: str) -> list[str]:
    return re.findall(r"[\u0600-\u06FF]+|[A-Za-z0-9_]+", normalize_arabic(text))

def rewrite_query(query: str) -> str:
    query = normalize_arabic(query)
    for src, dst in QUERY_SYNONYMS.items():
        query = re.sub(rf"\b{re.escape(normalize_arabic(src))}\b",
                       normalize_arabic(dst), query)
    return query

# Load ONCE at startup. Nothing is rebuilt inside retrieve().
with ARTICLES_PATH.open("r", encoding="utf-8") as f:
    ARTICLES: list[dict[str, Any]] = json.load(f)
with BM25_PATH.open("rb") as f:
    BM25: BM25Okapi = pickle.load(f)
EMBEDDINGS = np.load(EMBEDDINGS_PATH, mmap_mode="r")
if len(ARTICLES) != len(EMBEDDINGS):
    raise ValueError("legal_articles.json and embeddings.npy have different lengths")
ARTICLE_BY_ID = {i: a for i, a in enumerate(ARTICLES)}
MODEL = SentenceTransformer(MODEL_NAME)
if MODEL.get_sentence_embedding_dimension() != EMBEDDING_DIM:
    raise ValueError("BGE-M3 embedding dimension is not 1024")
QDRANT = QdrantClient(path=str(QDRANT_PATH))

def _filter(contract_type: str) -> Filter:
    return Filter(must=[
        FieldCondition(key="contract_types", match=MatchValue(value=contract_type)),
        FieldCondition(key="is_active", match=MatchValue(value=True)),
    ])

def _candidate_ids(contract_type: str) -> list[int]:
    points, _ = QDRANT.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=_filter(contract_type),
        limit=10000, with_payload=False, with_vectors=False
    )
    return [int(p.id) for p in points if str(p.id).isdigit()]

def dense_search(query: str, candidate_ids: list[int], top_k: int = 20) -> list[dict]:
    if not candidate_ids:
        return []
    vector = MODEL.encode([rewrite_query(query)], normalize_embeddings=True,
                          show_progress_bar=False)[0].tolist()
    allowed = set(candidate_ids)
    # Qdrant does semantic search; only IDs from the already-filtered
    # contract-type universe are accepted into the result.
    hits = QDRANT.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=Filter(must=[FieldCondition(
            key="is_active", match=MatchValue(value=True)
        )]),
        limit=max(top_k * 10, 100), with_payload=True, with_vectors=False
    )
    out = []
    for h in hits:
        try: idx = int(h.id)
        except (TypeError, ValueError): continue
        if idx not in allowed: continue
        out.append({"id":idx, "dense_score":float(h.score), "payload":h.payload or {}})
        if len(out) >= top_k: break
    return out

def bm25_search(query: str, candidate_ids: Iterable[int], top_k: int = 20) -> list[dict]:
    # CRITICAL FIX: BM25 ranks ONLY candidate_ids. It never searches all
    # articles for the final retrieval result.
    allowed = {int(i) for i in candidate_ids if 0 <= int(i) < len(ARTICLES)}
    if not allowed: return []
    scores = BM25.get_scores(tokenize_arabic(rewrite_query(query)))
    ranked = sorted(((i,float(scores[i])) for i in allowed),
                    key=lambda x:x[1], reverse=True)[:top_k]
    return [{"id":i, "bm25_score":s, "payload":ARTICLES[i]} for i,s in ranked]

def reciprocal_rank_fusion(dense_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    fused = {}
    for rank,r in enumerate(dense_results,1):
        x=fused.setdefault(int(r["id"]),{"id":int(r["id"]),"score":0.0})
        x["score"] += 1/(k+rank); x["dense_score"]=r.get("dense_score")
        x["payload"]=r.get("payload") or x.get("payload")
    for rank,r in enumerate(bm25_results,1):
        x=fused.setdefault(int(r["id"]),{"id":int(r["id"]),"score":0.0})
        x["score"] += 1/(k+rank); x["bm25_score"]=r.get("bm25_score")
        x["payload"]=x.get("payload") or r.get("payload")
    return sorted(fused.values(), key=lambda x:x["score"], reverse=True)

def format_result(r: dict) -> dict:
    a = r.get("payload") or ARTICLE_BY_ID[int(r["id"])]
    return {
        "score":r.get("score"), "dense_score":r.get("dense_score"),
        "bm25_score":r.get("bm25_score"),
        "law_name":a.get("law_name"), "law_number":a.get("law_number"),
        "law_year":a.get("law_year"), "article_number":a.get("article_number"),
        "article_label":a.get("article_label"), "chapter":a.get("chapter"),
        "binding_type":a.get("binding_type"), "contract_types":a.get("contract_types",[]),
        "needs_executive_regulation":a.get("needs_executive_regulation"),
        "superseded_by":a.get("superseded_by"), "is_active":a.get("is_active"),
        "start_page":a.get("start_page"), "end_page":a.get("end_page"),
        "source_page":a.get("source_page"), "text":a.get("text")
    }

def retrieve(clause_text: str, contract_type: str, top_k: int = 3) -> list[dict]:
    """Hybrid retrieval with ONE candidate universe shared by Qdrant and BM25.

    Pass 1: exact contract type.
    Pass 2: exact type + general only if pass 1 has fewer than top_k.
    Penalty articles are excluded for drafting.
    No final BM25 search over all_articles is performed.
    """
    if not clause_text or not str(clause_text).strip(): return []
    contract_type = str(contract_type).strip().lower()
    if contract_type not in SUPPORTED_CONTRACT_TYPES:
        raise ValueError(f"Unsupported contract_type: {contract_type}")
    top_k=max(1,int(top_k))

    exact_ids=_candidate_ids(contract_type)
    dense=dense_search(clause_text,exact_ids,max(top_k*5,20))
    sparse=bm25_search(clause_text,exact_ids,max(top_k*5,20))
    fused=reciprocal_rank_fusion(dense,sparse)
    fused=[r for r in fused if (r.get("payload") or {}).get("binding_type")!="penalty"]

    if len(fused)<top_k and contract_type!="general":
        general_ids=_candidate_ids("general")
        allowed_ids=list(dict.fromkeys(exact_ids+general_ids))
        dense=dense_search(clause_text,allowed_ids,max(top_k*5,20))
        sparse=bm25_search(clause_text,allowed_ids,max(top_k*5,20))
        fused=reciprocal_rank_fusion(dense,sparse)
        fused=[r for r in fused if (r.get("payload") or {}).get("binding_type")!="penalty"]

    allowed={contract_type,"general"}
    out=[]
    for r in fused:
        a=r.get("payload") or ARTICLE_BY_ID[int(r["id"])]
        types=set(a.get("contract_types") or [])
        if contract_type=="general":
            ok="general" in types
        else:
            ok=(contract_type in types) or (types=={"general"})
        if not ok: continue
        out.append(format_result(r))
        if len(out)>=top_k: break
    return out

def build_legal_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[Legal Source {i}]\nLaw: {r.get('law_name')}\n"
        f"Article: {r.get('article_number')}\nText: {r.get('text')}"
        for i,r in enumerate(results,1)
    )

__all__=["retrieve","build_legal_context","normalize_arabic",
         "tokenize_arabic","rewrite_query"]
