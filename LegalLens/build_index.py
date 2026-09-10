# ============================================================
# LegalLens — Build Index from legal_corpus.pkl
#
# Input:
#   data/legal_corpus.pkl
#
# Output:
#   index/qdrant_export/
#       collection_config.json
#       legallens_legal_only_points.json
#
# Embedding:
#   BAAI/bge-m3
#
# Similarity:
#   Cosine
# ============================================================

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

INDEX_DIR = (
    BASE_DIR
    / "index"
    / "qdrant_export"
)

CORPUS_PATH = (
    DATA_DIR / "legal_corpus.pkl"
)

OUTPUT_POINTS = (
    INDEX_DIR
    / "legallens_legal_only_points.json"
)

OUTPUT_CONFIG = (
    INDEX_DIR
    / "collection_config.json"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = os.getenv(
    "LEGALLENS_EMBEDDING_MODEL",
    "BAAI/bge-m3"
)

BATCH_SIZE = 8

EXPECTED_DIMENSION = 1024

COLLECTION_NAME = (
    "legallens_legal_only"
)


# ============================================================
# LOAD CORPUS
# ============================================================

def load_corpus():

    if not CORPUS_PATH.exists():

        raise FileNotFoundError(
            f"""
❌ legal_corpus.pkl not found:

{CORPUS_PATH}

Expected:

LegalLens/
├── data/
│   └── legal_corpus.pkl
└── build_index.py
"""
        )

    print(
        "📂 Loading legal_corpus.pkl..."
    )

    with open(
        CORPUS_PATH,
        "rb"
    ) as f:

        corpus = pickle.load(f)

    if not isinstance(
        corpus,
        list
    ):

        raise ValueError(
            "❌ legal_corpus.pkl must contain a list."
        )

    print(
        f"Loaded records: {len(corpus)}"
    )

    return corpus


# ============================================================
# PREPARE LEGAL ARTICLES
# ============================================================

def prepare_documents(corpus):

    documents = []

    seen = set()

    for item in corpus:

        if not isinstance(
            item,
            dict
        ):
            continue

        # ----------------------------------------------------
        # Legal articles only
        # ----------------------------------------------------

        doc_type = item.get(
            "doc_type",
            "legal_article"
        )

        if doc_type != "legal_article":
            continue

        # ----------------------------------------------------
        # Active articles only
        # ----------------------------------------------------

        if item.get(
            "is_active",
            True
        ) is False:

            continue

        law_id = str(
            item.get(
                "law_id",
                ""
            )
        ).strip()

        law_name = str(
            item.get(
                "law_name",
                ""
            )
        ).strip()

        article = str(
            item.get(
                "article",
                item.get(
                    "article_number",
                    ""
                )
            )
        ).strip()

        text = str(
            item.get(
                "text",
                item.get(
                    "article_text",
                    ""
                )
            )
        ).strip()

        source = str(
            item.get(
                "source",
                ""
            )
        ).strip()

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        if not law_id:
            continue

        if not article:
            continue

        if not text:
            continue

        # ----------------------------------------------------
        # Duplicate check
        # ----------------------------------------------------

        key = (
            law_id,
            article
        )

        if key in seen:

            raise ValueError(
                f"❌ Duplicate article:\n"
                f"{law_id} / {article}"
            )

        seen.add(key)

        documents.append(
            {
                "law_id":
                    law_id,

                "law_name":
                    law_name,

                "article_number":
                    article,

                "article_text":
                    text,

                "is_active":
                    True,

                "source":
                    source
            }
        )

    if not documents:

        raise ValueError(
            "❌ No legal articles found."
        )

    return documents


# ============================================================
# BUILD EMBEDDING TEXT
# ============================================================

def build_embedding_text(
    document
):

    return (
        f"القانون: "
        f"{document['law_name']}\n"
        f"المادة: "
        f"{document['article_number']}\n"
        f"{document['article_text']}"
    )


# ============================================================
# BUILD INDEX
# ============================================================

def build_index():

    print("=" * 90)
    print("LEGALLENS — BUILD FINAL LEGAL INDEX")
    print("=" * 90)

    # --------------------------------------------------------
    # 1. Load corpus
    # --------------------------------------------------------

    corpus = load_corpus()

    # --------------------------------------------------------
    # 2. Prepare legal articles
    # --------------------------------------------------------

    documents = prepare_documents(
        corpus
    )

    print(
        f"Legal articles: {len(documents)}"
    )

    # --------------------------------------------------------
    # 3. Create output directory
    # --------------------------------------------------------

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 4. Load BGE-M3
    # --------------------------------------------------------

    print(
        "\n🤖 Loading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    # --------------------------------------------------------
    # 5. Prepare texts
    # --------------------------------------------------------

    texts = [
        build_embedding_text(doc)
        for doc in documents
    ]

    # --------------------------------------------------------
    # 6. Generate embeddings
    # --------------------------------------------------------

    print(
        "\n🧠 Generating embeddings..."
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # 7. Validate embeddings
    # --------------------------------------------------------

    if embeddings.ndim != 2:

        raise ValueError(
            f"❌ Invalid embedding shape: "
            f"{embeddings.shape}"
        )

    if embeddings.shape[1] != EXPECTED_DIMENSION:

        raise ValueError(
            f"❌ Expected {EXPECTED_DIMENSION} dimensions, "
            f"got {embeddings.shape[1]}"
        )

    if not np.isfinite(
        embeddings
    ).all():

        raise ValueError(
            "❌ Embeddings contain NaN or Inf."
        )

    print(
        f"Embedding matrix: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # 8. Build Qdrant-compatible points
    # --------------------------------------------------------

    print(
        "\n📦 Building points..."
    )

    points = []

    for idx, (
        document,
        vector
    ) in enumerate(
        zip(
            documents,
            embeddings
        )
    ):

        points.append(
            {
                "id": idx,

                "vector":
                    vector.tolist(),

                "payload":
                    {
                        "law_id":
                            document["law_id"],

                        "law_name":
                            document["law_name"],

                        "article":
                            document[
                                "article_number"
                            ],

                        "text":
                            document[
                                "article_text"
                            ],

                        "source":
                            document.get(
                                "source",
                                ""
                            ),

                        "doc_type":
                            "legal_article",

                        "is_active":
                            True
                    }
            }
        )

    # --------------------------------------------------------
    # 9. Save points
    # --------------------------------------------------------

    print(
        "\n💾 Saving Qdrant export..."
    )

    with open(
        OUTPUT_POINTS,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            points,
            f,
            ensure_ascii=False,
            separators=(
                ",",
                ":"
            )
        )

    # --------------------------------------------------------
    # 10. Save configuration
    # --------------------------------------------------------

    config = {

        "collection_name":
            COLLECTION_NAME,

        "embedding_model":
            MODEL_NAME,

        "vector_size":
            EXPECTED_DIMENSION,

        "distance":
            "Cosine",

        "documents":
            len(documents),

        "source":
            "data/legal_corpus.pkl",

        "normalized_embeddings":
            True,

        "doc_type":
            "legal_article",

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat()
    }

    with open(
        OUTPUT_CONFIG,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # 11. Final report
    # --------------------------------------------------------

    points_size = (
        OUTPUT_POINTS.stat().st_size
        / (1024 ** 2)
    )

    print("\n" + "=" * 90)
    print("✅ LEGALLENS INDEX BUILD COMPLETED")
    print("=" * 90)

    print(
        f"Articles       : {len(documents)}"
    )

    print(
        f"Vectors        : {len(points)}"
    )

    print(
        f"Dimension      : {embeddings.shape[1]}"
    )

    print(
        f"Model          : {MODEL_NAME}"
    )

    print(
        "Similarity     : Cosine"
    )

    print(
        f"Points file    : {OUTPUT_POINTS}"
    )

    print(
        f"Points size    : {points_size:.2f} MB"
    )

    print(
        f"Config file    : {OUTPUT_CONFIG}"
    )

    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_index()
