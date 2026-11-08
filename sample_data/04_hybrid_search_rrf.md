# Hybrid search, RRF, and query strategies

## Hybrid search
Hybrid search combines dense semantic retrieval with sparse lexical retrieval. Dense search matches meaning and paraphrases; BM25-style search matches exact terms and identifiers. The system merges and deduplicates candidates from both.

## Reciprocal Rank Fusion (RRF)
Dense and BM25 scores use different scales, so adding their raw scores can be misleading. RRF combines ranked lists using each result's position:

RRF(document) = sum of 1 / (k + rank) across ranking lists.

Here rank is the document's position in one list and k is a smoothing constant (often 60). A document ranked highly by multiple retrievers receives a stronger combined score. RRF uses ranks, so it does not require raw scores to be calibrated.

## Typical hybrid pipeline
1. Run dense retrieval and BM25 separately.
2. Retrieve a broad candidate set from each.
3. Merge and deduplicate candidates.
4. Fuse rankings with RRF.
5. Optionally rerank the fused candidates.
6. Keep the final top-k passages for generation.

## Query strategies
- **Query rewriting:** turn an unclear or conversational question into a clearer standalone search query using context.
- **Multi-query:** generate alternative phrasings, search each, then merge results. This may improve recall but adds latency and noise.
- **Query decomposition:** break a complex question into smaller subquestions, search each, and combine evidence.
- **Metadata filtering:** restrict results by date, category, source, or permissions.
- **HyDE:** generate a hypothetical answer-like passage and use its embedding as a search query. It can bridge wording gaps, but the hypothetical text may contain errors.
- **Parent-child retrieval:** search small chunks but return a larger parent section.
- **Contextual retrieval:** add a short description of a chunk's position in its source before indexing, helping ambiguous chunks make sense.

Sources:
- https://www.pinecone.io/learn/retrieval-augmented-generation/
- https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval
