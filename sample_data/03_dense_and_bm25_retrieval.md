# Dense retrieval, sparse retrieval, and BM25

Retrieval is the search stage of RAG. It finds passages likely to help answer the question. A language model cannot use evidence that retrieval failed to find.

## Dense retrieval
Dense retrieval uses embeddings. The system converts each document chunk and the user's query into vectors, then compares them with a similarity measure such as cosine similarity or dot product.

Example: a question about "finding semantically related passages" may retrieve a chunk about vector similarity even when the wording differs.

Strengths: matches paraphrases and related meanings. Weaknesses: can miss exact identifiers, rare names, error codes, and technical strings. Similarity is not the same as factual relevance.

## Sparse / lexical retrieval
Sparse retrieval represents text using weighted terms. Traditional keyword search is a familiar example. It is strong when query words also appear in the document.

## BM25
BM25 is a common lexical ranking algorithm. It scores documents using query-term frequency, how rare each term is across the collection, and document-length normalisation.

- A matching query term increases a document's score.
- Rare terms usually provide stronger evidence than common terms.
- Repeating a term helps, but with diminishing returns.
- Length normalisation prevents long documents from automatically winning.

BM25 is useful for exact technical terms, names, acronyms, identifiers, and error messages.

## Dense versus BM25
- Paraphrased natural-language question: dense retrieval often helps.
- Exact API name or error code: BM25 often helps.
- Natural-language question containing domain-specific terms: try hybrid retrieval.

Neither method is always best. Evaluate using realistic questions.

## Top-k
Top-k is the number of passages returned. Small k may miss useful evidence (low recall). Large k may add noise, latency, and cost. Tune it with evaluation rather than guessing.

Source: https://www.pinecone.io/learn/series/rag/
