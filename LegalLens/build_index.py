# ============================================================
# LegalLens — Build Legal Retrieval Index
#
# Input:
#   data/laws.jsonl
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
#   Cosine similarity
# ============================================================

from __future__ import annotations

import json
import os
import shutil
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

LAWS_PATH = DATA_DIR / "laws.jsonl"

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

EMBEDDING_DIMENSION = 1024

COLLECTION_NAME = (
    "legallens_legal_only"
)


# ============================================================
# VALIDATION
# ============================================================

REQUIRED_FIELDS = {
    "law_id",
    "law_name",
    "article_number",
    "article_text",
    "is_active",
}


# ============================================================
# LOAD LAWS
# ============================================================

def load_laws():

    if not LAWS_PATH.exists():

        raise FileNotFoundError(
            f"""
❌ laws.jsonl not found:

{LAWS_PATH}

Expected structure:

LegalLens/
├── data/
│   └── laws.jsonl
└── build_index.py
"""
        )

    documents = []

    with open(
        LAWS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                item = json.loads(line)

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"❌ Invalid JSON at line "
                    f"{line_number}: {e}"
                )

            missing = (
                REQUIRED_FIELDS
                - set(item.keys())
            )

            if missing:

                raise ValueError(
                    f"❌ Missing fields at line "
                    f"{line_number}: "
                    f"{sorted(missing)}"
                )

            documents.append(item)

    return documents


# ============================================================
# FILTER ACTIVE LEGAL ARTICLES
# ============================================================

def prepare_documents(documents):

    legal_documents = []

    seen = set()

    for item in documents:

        # ----------------------------------------------------
        # Only active articles
        # ----------------------------------------------------

        is_active = item.get(
            "is_active",
            True
        )

        if not is_active:
            continue

        law_id = str(
            item["law_id"]
        ).strip()

        article_number = str(
            item["article_number"]
        ).strip()

        article_text = str(
            item["article_text"]
        ).strip()

        law_name = str(
            item["law_name"]
        ).strip()

        if not law_id:
            continue

        if not article_number:
            continue

        if not article_text:
            continue

        # ----------------------------------------------------
        # Unique article ID
        # ----------------------------------------------------

        key = (
            law_id,
            article_number
        )

        if key in seen:

            raise ValueError(
                "❌ Duplicate article detected: "
                f"{law_id} / {article_number}"
            )

        seen.add(key)

        legal_documents.append(
            {
                "law_id": law_id,

                "law_name": law_name,

                "article_number":
                    article_number,

                "article_text":
                    article_text,

                "is_active": True,

                "source":
                    item.get(
                        "source",
                        ""
                    ),
            }
        )

    return legal_documents


# ============================================================
# BUILD EMBEDDING TEXT
# ============================================================

def build_embedding_text(
    document
):

    # We intentionally embed the legal article itself.
    #
    # Including the law name helps distinguish articles
    # belonging to different laws.

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

    print("=" * 80)
    print("LEGALLENS — BUILD FINAL LEGAL INDEX")
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\n📂 Loading laws.jsonl...")

    all_documents = load_laws()

    print(
        f"Loaded records: {len(all_documents)}"
    )

    # --------------------------------------------------------
    # Prepare active legal articles
    # --------------------------------------------------------

    documents = prepare_documents(
        all_documents
    )

    print(
        f"Active legal articles: "
        f"{len(documents)}"
    )

    if not documents:

        raise ValueError(
            "❌ No active legal articles found."
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load BGE-M3
    # --------------------------------------------------------

    print("\n🤖 Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    # --------------------------------------------------------
    # Prepare texts
    # --------------------------------------------------------

    texts = [
        build_embedding_text(doc)
        for doc in documents
    ]

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print("\n🧠 Generating embeddings...")

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
    # Validate dimensions
    # --------------------------------------------------------

    if embeddings.ndim != 2:

        raise ValueError(
            "❌ Invalid embedding matrix shape: "
            f"{embeddings.shape}"
        )

    if embeddings.shape[1] != EMBEDDING_DIMENSION:

        raise ValueError(
            "❌ Unexpected embedding dimension: "
            f"{embeddings.shape[1]} "
            f"(expected {EMBEDDING_DIMENSION})"
        )

    print(
        f"Embedding matrix: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------------
    # Build exported points
    # --------------------------------------------------------

    print("\n📦 Building Qdrant-compatible points...")

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

        point = {

            "id": idx,

            "vector": vector.tolist(),

            "payload": {

                "law_id":
                    document["law_id"],

                "law_name":
                    document["law_name"],

                "article":
                    document["article_number"],

                "text":
                    document["article_text"],

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

        points.append(point)

    # --------------------------------------------------------
    # Save points
    # --------------------------------------------------------

    print("\n💾 Saving points...")

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
    # Collection configuration
    # --------------------------------------------------------

    config = {

        "collection_name":
            COLLECTION_NAME,

        "distance":
            "Cosine",

        "vector_size":
            EMBEDDING_DIMENSION,

        "embedding_model":
            MODEL_NAME,

        "documents":
            len(documents),

        "source":
            "data/laws.jsonl",

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "normalized_embeddings":
            True,

        "doc_type":
            "legal_article"
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
    # Final validation
    # --------------------------------------------------------

    points_size = (
        OUTPUT_POINTS.stat().st_size
        / (1024 ** 2)
    )

    print("\n" + "=" * 80)
    print("✅ FINAL INDEX BUILD COMPLETED")
    print("=" * 80)

    print(
        f"Legal articles : {len(documents)}"
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
        f"Similarity      : Cosine"
    )

    print(
        f"Points file     : {OUTPUT_POINTS}"
    )

    print(
        f"Points size     : {points_size:.2f} MB"
    )

    print(
        f"Config file     : {OUTPUT_CONFIG}"
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_index()
