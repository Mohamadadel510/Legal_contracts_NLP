# ============================================================
# LegalLens - Final Retriever
# BAAI/bge-m3 + exported Qdrant vectors
# ============================================================

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Expected structure:
#
# rag/
# ├── modules/
# │   └── retriever.py
# │
# └── index/
#     └── qdrant_export/
#         ├── collection_config.json
#         └── legallens_legal_only_points.json

INDEX_DIR = BASE_DIR / "index" / "qdrant_export"

POINTS_PATH = (
    INDEX_DIR / "legallens_legal_only_points.json"
)

CONFIG_PATH = (
    INDEX_DIR / "collection_config.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = os.getenv(
    "LEGALLENS_EMBEDDING_MODEL",
    "BAAI/bge-m3"
)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_TOP_K = 3

# None means no filtering.
# Example:
# SIMILARITY_THRESHOLD = 0.45
#
# We leave it None initially because the correct threshold
# should be calibrated on validation data.
SIMILARITY_THRESHOLD = None


# ============================================================
# CACHED GLOBAL RESOURCES
# ============================================================

_model: SentenceTransformer | None = None

_vectors: np.ndarray | None = None

_metadata: list[dict[str, Any]] | None = None

_initialized = False


# ============================================================
# LOAD MODEL
# ============================================================

def _load_model() -> SentenceTransformer:

    global _model

    if _model is None:

        _model = SentenceTransformer(
            MODEL_NAME
        )

    return _model


# ============================================================
# LOAD EXPORTED QDRANT DATA
# ============================================================

def _load_points() -> None:

    global _vectors
    global _metadata
    global _initialized

    if _initialized:
        return

    if not POINTS_PATH.exists():

        raise FileNotFoundError(
            f"Final RAG index not found:\n"
            f"{POINTS_PATH}"
        )

    print(
        "Loading LegalLens final index..."
    )

    with open(
        POINTS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        points = json.load(f)

    if not points:

        raise ValueError(
            "The Qdrant export contains no points."
        )

    vectors = []
    metadata = []

    for point in points:

        vector = point.get(
            "vector"
        )

        payload = point.get(
            "payload",
            {}
        )

        if vector is None:

            continue

        vectors.append(
            vector
        )

        metadata.append(
            {
                "id": point.get("id"),

                "law_id": payload.get(
                    "law_id",
                    ""
                ),

                "law_name": payload.get(
                    "law_name",
                    ""
                ),

                "article_number": str(
                    payload.get(
                        "article",
                        payload.get(
                            "article_number",
                            ""
                        )
                    )
                ),

                "article_text": payload.get(
                    "text",
                    payload.get(
                        "article_text",
                        ""
                    )
                ),

                "source": payload.get(
                    "source",
                    ""
                ),

                "doc_type": payload.get(
                    "doc_type",
                    "legal_article"
                )
            }
        )

    if not vectors:

        raise ValueError(
            "No vectors found in the Qdrant export."
        )

    _vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    _metadata = metadata

    # --------------------------------------------------------
    # Normalize vectors
    #
    # The original Final RAG used normalized BGE-M3 vectors
    # with cosine similarity.
    # --------------------------------------------------------

    norms = np.linalg.norm(
        _vectors,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    _vectors = (
        _vectors / norms
    )

    _initialized = True

    print(
        f"✅ Loaded {_vectors.shape[0]} legal vectors"
    )

    print(
        f"✅ Vector dimension: {_vectors.shape[1]}"
    )


# ============================================================
# INITIALIZE
# ============================================================

def initialize() -> None:
    """
    Load the model and final legal index once.

    Call this when the application starts.
    """

    _load_model()

    _load_points()


# ============================================================
# EMBED QUERY
# ============================================================

def _embed_query(
    query: str
) -> np.ndarray:

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
# RETRIEVE
# ============================================================

def retrieve(
    clause_text: str,
    contract_type: str,
    top_k: int = 3
) -> list[dict]:
    """
    Retrieve the most relevant Egyptian legal articles.

    Parameters
    ----------
    clause_text:
        Contract clause or legal question.

    contract_type:
        Contract type supplied by the application.

        NOTE:
        It is intentionally NOT used as a strict filter.
        The previous experiments showed that contract-type
        classification can produce false positives.

    top_k:
        Number of legal articles to return.

    Returns
    -------
    list[dict]

    Example
    -------
    [
        {
            "law_id": "labor_law_14_2025",
            "law_name": "قانون العمل رقم 14 لسنة 2025",
            "article_number": "89",
            "article_text": "...",
            "score": 0.7421,
            "source": "..."
        }
    ]
    """

    # --------------------------------------------------------
    # Validate query
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

    top_k = int(top_k)

    # --------------------------------------------------------
    # Load resources ONCE
    # --------------------------------------------------------

    initialize()

    if (
        _vectors is None
        or _metadata is None
    ):

        raise RuntimeError(
            "LegalLens retriever is not initialized."
        )

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_vector = _embed_query(
        clause_text
    )

    # --------------------------------------------------------
    # Cosine similarity
    #
    # Both query and document vectors are normalized.
    # Therefore:
    #
    # cosine_similarity = dot_product
    # --------------------------------------------------------

    scores = (
        _vectors @ query_vector[0]
    )

    # --------------------------------------------------------
    # Get top candidates
    # --------------------------------------------------------

    k = min(
        top_k,
        len(scores)
    )

    candidate_indices = np.argpartition(
        -scores,
        k - 1
    )[:k]

    # Sort candidates by score
    candidate_indices = candidate_indices[
        np.argsort(
            -scores[candidate_indices]
        )
    ]

    # --------------------------------------------------------
    # Build final results
    # --------------------------------------------------------

    results = []

    for rank, idx in enumerate(
        candidate_indices,
        start=1
    ):

        score = float(
            scores[idx]
        )

        # Optional threshold
        if (
            SIMILARITY_THRESHOLD is not None
            and score < SIMILARITY_THRESHOLD
        ):
            continue

        item = _metadata[idx]

        results.append(
            {
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
        )

    return results


# ============================================================
# HEALTH CHECK
# ============================================================

def health_check() -> dict:

    initialize()

    return {
        "status": "ok",

        "embedding_model":
            MODEL_NAME,

        "index":
            "legallens_legal_only",

        "documents":
            len(_metadata)
            if _metadata
            else 0,

        "vector_dimension":
            int(_vectors.shape[1])
            if _vectors is not None
            else 0,

        "similarity":
            "cosine",

        "default_top_k":
            DEFAULT_TOP_K,

        "similarity_threshold":
            SIMILARITY_THRESHOLD
    }