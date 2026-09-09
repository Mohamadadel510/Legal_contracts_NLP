# LegalLens Retriever

## 1. Purpose

This module provides the retrieval layer for the LegalLens contract-generation pipeline.

It retrieves relevant Egyptian legal articles for a requested contract clause using a **Hybrid RAG** approach:

1. Dense semantic retrieval with `BAAI/bge-m3`
2. Sparse keyword retrieval with BM25
3. Reciprocal Rank Fusion (RRF)
4. Contract-type filtering
5. Fallback to general legal provisions when needed

The retriever is designed to load the index and models **once at application startup**, rather than rebuilding the index for every request.

---

## 2. Public API

The main function is:

```python
retrieve(
    clause_text: str,
    contract_type: str,
    top_k: int = 3
) -> list[dict]
```

### Parameters

- `clause_text`: The contract clause or requirement that needs legal support.
- `contract_type`: The type of contract being drafted.
- `top_k`: Number of legal articles to return. Default: `3`.

### Supported contract types

```text
residential
agricultural
commercial
employment
company
sale
supply
service
consumer
general
```

---

## 3. Retrieval behavior

The retriever first searches for articles explicitly tagged with the requested contract type.

If fewer than `top_k` suitable articles are found, it can fall back to articles tagged as `general`.

This is intentional: a residential contract should not receive agricultural-only provisions simply because they are semantically similar.

For the drafting flow, penalty clauses are excluded from the returned drafting context.

The retrieval pipeline is:

```text
Clause
  ↓
Arabic normalization / query rewriting
  ↓
Dense search (BGE-M3)
  +
BM25 search
  ↓
RRF fusion
  ↓
Contract-type filtering
  ↓
Penalty exclusion for drafting
  ↓
Top-k legal articles
```

---

## 4. Returned fields

Each result is returned as a dictionary containing the legal article and retrieval metadata.

Typical fields are:

```text
score
dense_score
bm25_score
law_name
law_number
law_year
article_number
article_label
binding_type
contract_types
needs_executive_regulation
superseded_by
is_active
start_page
end_page
source_page
text
```

### Important fields for the Drafter

At minimum, the Drafter should use:

- `law_name`
- `article_number`
- `text`

The additional metadata allows the final generated contract or internal trace to show where the legal support came from.

### Scores

- `dense_score`: semantic similarity score from the dense retrieval stage.
- `bm25_score`: lexical relevance score from BM25.
- `score`: final fused ranking score used by the hybrid retriever.

The final `score` is an RRF-based ranking score, not a probability and should not be interpreted as a percentage of legal correctness.

---

## 5. Embedding model

The dense retrieval model is:

```text
BAAI/bge-m3
```

Embedding dimension:

```text
1024
```

Embeddings are normalized before vector search.

The model is loaded once when the retriever module starts.

### Internet requirement

The model may need internet access on the **first run** if it is not already cached locally.

After the model has been downloaded and cached, the retriever can run without internet access, provided all required model files and local indexes are available.

For a fully offline deployment, bundle/cache the model in the deployment environment before running the application.

---

## 6. Index components

The retriever expects the project to contain:

```text
LegalLens/
├── modules/
│   └── retriever.py
│
├── data/
│   ├── legal_articles.json
│   ├── bm25.pkl
│   └── embeddings.npy
│
└── index/
    └── legal_rag_qdrant/
```

The local Qdrant collection is:

```text
egyptian_legal_articles_contract_types
```

The index contains the legal article text and metadata required for filtered retrieval.

---

## 7. Expected source data

`build_index.py` accepts:

```text
data/laws.jsonl
```

Each JSONL line represents one legal article.

The normalized schema used by the index is:

```json
{
  "law_id": "...",
  "law_name": "...",
  "law_number": "...",
  "law_year": "...",
  "article_number": "...",
  "article_label": "...",
  "chapter": "...",
  "binding_type": "...",
  "needs_executive_regulation": false,
  "superseded_by": null,
  "is_active": true,
  "start_page": null,
  "end_page": null,
  "text": "...",
  "contract_types": ["general"]
}
```

The most important fields are:

```text
law_id
law_name
article_number
text
```

The remaining fields provide filtering, traceability, and legal metadata.

### Article-level indexing

The atomic retrieval unit is the **whole legal article**.

The system should not arbitrarily split one legal article into unrelated token chunks when building the legal index.

Special article labels such as amended/repeated articles should be represented consistently in the metadata.

---

## 8. Contract-type classification

`build_index.py` can preserve supplied `contract_types`.

When a record does not already contain contract types, the index builder applies a conservative rule-based classifier.

The current MVP classification supports:

```text
residential
agricultural
commercial
employment
company
sale
supply
service
consumer
general
```

`general` is used for provisions that can apply across multiple contract types.

The classifier is intentionally conservative because a legal provision should not be assigned to a contract type merely because it contains a generic word such as "lease" or "contract".

For production use, contract-type assignments should be legally reviewed.

---

## 9. Rebuilding the index

When new laws or updated legal articles are added, rebuild the retrieval artifacts with:

```bash
python build_index.py \
  --input data/laws.jsonl \
  --data-dir data \
  --index-dir index
```

Optional parameters include:

```text
--model
--collection
```

The rebuild process generates/updates:

```text
data/legal_articles.json
data/legal_articles.csv
data/article_metadata.json
data/bm25.pkl
data/embeddings.npy
index/legal_rag_qdrant/
```

### Important

Do not run the application/retriever against the local Qdrant directory while rebuilding the same local Qdrant index.

Local Qdrant uses a storage lock. If another Qdrant client is already using:

```text
index/legal_rag_qdrant/
```

the rebuild may fail with an `AlreadyLocked` error.

For a production multi-process deployment, a Qdrant server/cloud deployment is preferable to embedded local storage.

---

## 10. Example usage

```python
from modules.retriever import retrieve

results = retrieve(
    clause_text="يلتزم المستأجر بسداد الأجرة في المواعيد المتفق عليها",
    contract_type="residential",
    top_k=3,
)

for result in results:
    print(result["law_name"])
    print(result["article_number"])
    print(result["text"])
    print(result["score"])
```

---

## 11. Passing results to the Drafter

The returned results can be converted into a legal context before sending them to the drafting LLM.

Example:

```python
results = retrieve(
    clause_text=clause_text,
    contract_type="residential",
    top_k=3,
)

context = build_legal_context(results)
```

The Drafter should use the retrieved legal articles as supporting legal context and should preserve article/law references where required by the application.

A deterministic post-check can be used after generation to verify that any quoted legal text actually exists in the original retrieved article.

---

## 12. Design decisions

### Why Hybrid RAG?

Legal retrieval benefits from both:

- semantic similarity for meaning and paraphrases
- lexical matching for exact legal terminology

Combining dense retrieval and BM25 through RRF provides a more robust retrieval layer than relying on either method alone.

### Why contract-type filtering?

Pure semantic similarity can return legally related but inappropriate provisions.

For example, an agricultural lease provision may be semantically close to a residential lease request. Contract-type filtering prevents that provision from being selected merely because the wording is similar.

### Why return multiple articles?

A clause may be governed by more than one legal provision. Returning the top 3 articles gives the Drafter a small, focused legal context without flooding the prompt with the entire legal database.

---

## 13. Current MVP limitations

1. Contract-type tags are currently an MVP classification layer and should receive legal validation before production.
2. Source parsing quality directly affects retrieval quality.
3. Duplicate article records should be cleaned during data preparation.
4. `score` represents retrieval ranking, not legal validity.
5. Local Qdrant storage is appropriate for a single-process/demo setup but is not the preferred architecture for concurrent production workloads.
6. BGE-M3 must be available locally for fully offline runtime operation.

---

## 14. Recommended handoff contract

The Drafter team can treat the retriever as a black-box component:

```python
results = retrieve(
    clause_text,
    contract_type,
    top_k=3
)
```

They do not need to know how embeddings, BM25, Qdrant, or RRF work internally.

They can rely on the returned schema, especially:

```text
law_name
article_number
text
score
dense_score
bm25_score
```

This keeps retrieval implementation separate from contract-generation logic and allows the index to be rebuilt independently when new legal sources are added.
