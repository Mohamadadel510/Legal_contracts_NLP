# modules/retriever.py
# ============================================================
# LegalLens - Final Legal Retriever
# BAAI/bge-m3 + FAISS
# ============================================================

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = BASE_DIR / "index"
DATA_DIR = BASE_DIR / "data"

FAISS_INDEX_PATH = INDEX_DIR / "legal_articles.faiss"
METADATA_PATH = INDEX_DIR / "article_metadata.json"

# Model can be overridden with an environment variable.
# For completely offline deployment, point this to a local
# copy of BAAI/bge-m3.
MODEL_PATH = os.getenv(
    "LEGALLENS_EMBEDDING_MODEL",
    "BAAI/bge-m3"
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_TOP_K = 3

# Optional similarity threshold.
# None = do not filter by score.
#
# FAISS IndexFlatIP with normalized vectors gives cosine
# similarity.
SIMILARITY_THRESHOLD = None


# ============================================================
# GLOBAL SINGLETONS
# ============================================================

_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_metadata: list[dict[str, Any]] | None = None


# ============================================================
# MODEL
# ============================================================

def _load_model() -> SentenceTransformer:
    """
    Load the embedding model once.

    The model is NOT loaded for every retrieve() call.
    """

    global _model

    if _model is None:

        _model = SentenceTransformer(
            MODEL_PATH
        )

    return _model


# ============================================================
# INDEX
# ============================================================

def _load_index() -> faiss.Index:
    """
    Load the FAISS index once.
    """

    global _index

    if _index is None:

        if not FAISS_INDEX_PATH.exists():

            raise FileNotFoundError(
                f"FAISS index not found: {FAISS_INDEX_PATH}\n"
                "Run build_index.py first."
            )

        _index = faiss.read_index(
            str(FAISS_INDEX_PATH)
        )

    return _index


# ============================================================
# METADATA
# ============================================================

def _load_metadata() -> list[dict[str, Any]]:
    """
    Load article metadata once.
    """

    global _metadata

    if _metadata is None:

        if not METADATA_PATH.exists():

            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_PATH}\n"
                "Run build_index.py first."
            )

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        # Support either:
        #   [...]
        # or
        #   {"articles": [...]}
        if isinstance(data, dict):

            if "articles" in data:

                data = data["articles"]

            else:

                raise ValueError(
                    "Metadata JSON must contain an 'articles' list."
                )

        if not isinstance(data, list):

            raise ValueError(
                "Metadata must be a list of article records."
            )

        _metadata = data

    return _metadata


# ============================================================
# INITIALIZE
# ============================================================

def _initialize() -> None:
    """
    Load everything once.

    This function is intentionally separate from retrieve()
    so the index/model are never rebuilt per request.
    """

    _load_model()
    _load_index()
    _load_metadata()


# ============================================================
# QUERY EMBEDDING
# ============================================================

def _embed_query(query: str) -> np.ndarray:
    """
    Encode one Arabic legal query and normalize it.
    """

    model = _load_model()

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embedding.astype(
        np.float32
    )


# ============================================================
# PUBLIC RETRIEVER
# ============================================================

def retrieve(
    clause_text: str,
    contract_type: str,
    top_k: int = DEFAULT_TOP_K
) -> list[dict]:
    """
    Retrieve the most relevant Egyptian legal articles.

    Parameters
    ----------
    clause_text:
        The contract clause / legal query.

    contract_type:
        Contract type supplied by the application.
        It is intentionally NOT used as a strict filter.

    top_k:
        Number of legal articles to return.

    Returns
    -------
    list[dict]

    Example:
    [
        {
            "law_id": "...",
            "law_name": "...",
            "article_number": "89",
            "article_text": "...",
            "score": 0.7421,
            "source": "..."
        }
    ]
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not isinstance(
        clause_text,
        str
    ):

        raise TypeError(
            "clause_text must be a string."
        )

    clause_text = clause_text.strip()

    if not clause_text:

        return []

    if top_k <= 0:

        return []

    # Prevent unreasonable requests.
    top_k = int(top_k)

    # --------------------------------------------------------
    # Load cached resources
    # --------------------------------------------------------

    _initialize()

    model = _model
    index = _index
    metadata = _metadata

    if model is None or index is None or metadata is None:

        raise RuntimeError(
            "LegalLens retriever failed to initialize."
        )

    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    query_vector = _embed_query(
        clause_text
    )

    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    scores, indices = index.search(
        query_vector,
        min(top_k, index.ntotal)
    )

    results = []

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        # FAISS uses -1 for missing results.
        if idx < 0:
            continue

        if idx >= len(metadata):
            continue

        score = float(score)

        # Optional threshold.
        if (
            SIMILARITY_THRESHOLD is not None
            and score < SIMILARITY_THRESHOLD
        ):
            continue

        item = metadata[idx]

        result = {
            "law_id": item.get(
                "law_id",
                ""
            ),

            "law_name": item.get(
                "law_name",
                ""
            ),

            "article_number": str(
                item.get(
                    "article_number",
                    ""
                )
            ),

            "article_text": item.get(
                "article_text",
                ""
            ),

            "score": round(
                score,
                6
            ),

            "source": item.get(
                "source",
                ""
            )
        }

        results.append(
            result
        )

    return results


# ============================================================
# OPTIONAL STARTUP FUNCTION
# ============================================================

def initialize() -> None:
    """
    Explicitly preload the model, FAISS index and metadata.

    Recommended to call once when the application starts.
    """

    _initialize()


# ============================================================
# HEALTH CHECK
# ============================================================

def health_check() -> dict:
    """
    Return retriever status.
    """

    _initialize()

    return {
        "status": "ok",
        "embedding_model": MODEL_PATH,
        "index_type": type(_index).__name__,
        "index_size": int(
            _index.ntotal
        ) if _index is not None else 0,
        "metadata_size": len(
            _metadata
        ) if _metadata is not None else 0,
        "top_k_default": DEFAULT_TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD
    }
