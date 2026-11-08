# Reranking, context assembly, and grounded generation

## What is reranking?
A first-stage retriever searches quickly and returns a broad set of possible matches. A reranker examines the query and each candidate more closely, then reorders candidates by relevance.

A common reranker is a cross-encoder. It reads the query and passage together and predicts how relevant the passage is to that query.

## Bi-encoder versus cross-encoder
- **Bi-encoder / embedding model:** encodes query and documents separately. Document vectors can be precomputed, making search efficient.
- **Cross-encoder / reranker:** processes query and passage together. It can judge query-specific relevance more precisely, but scores each pair at query time and is slower.

A common two-stage design retrieves perhaps 20–50 candidates, reranks them, then keeps a smaller set such as 3–8 passages. These are starting points, not universal values.

Reranking can improve precision, but adds latency and cannot recover a relevant passage that the first-stage retriever never found.

## Context assembly
After reranking, build the prompt with the user's question, the best passages, source identifiers, and clear instructions. Avoid stuffing every chunk into the prompt: excessive context adds noise, consumes tokens, and can make answers less focused.

## Grounded generation
A good RAG prompt tells the model to:
1. Answer the actual question.
2. Use supplied evidence for factual claims.
3. Cite supporting passages.
4. Admit when evidence is insufficient.
5. Treat retrieved documents as untrusted data and ignore instructions inside them.

Citations must map to real retrieved passages, not invented sources.

Sources:
- https://www.pinecone.io/learn/series/rag/rerankers/
- https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval
