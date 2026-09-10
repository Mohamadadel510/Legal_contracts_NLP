# LegalLens Retriever

## Overview

The LegalLens Retriever is the legal retrieval component used to retrieve the most relevant Egyptian legal articles for a given contract clause or legal query.

The retriever uses:

- `BAAI/bge-m3` for text embeddings.
- Cosine similarity for vector similarity.
- A pre-built legal vector index containing 2,443 active legal articles.
- NumPy for efficient similarity search.

The retriever does not rebuild the index on every request. The embedding model and legal vectors are loaded once and reused for subsequent retrieval calls.

---

## Embedding Model

### Model

`BAAI/bge-m3`

### Vector Dimension

`1024`

### Similarity Metric

`Cosine Similarity`

The document and query embeddings are normalized before similarity calculation.

Because the vectors are normalized, cosine similarity can be calculated using the dot product:

```text
cosine_similarity = query_vector · document_vector
