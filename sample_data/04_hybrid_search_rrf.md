# Hybrid search and Reciprocal Rank Fusion

Hybrid search combines results from different retrieval methods, commonly dense vector retrieval and sparse or lexical retrieval. The methods can retrieve different but useful evidence, so combining them may improve coverage.

## Why score fusion is not always straightforward

Dense similarity scores and BM25 scores are produced by different functions and can have different numerical ranges. Directly adding raw scores can let one retriever dominate for reasons unrelated to relevance.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) combines ranked lists using rank positions rather than raw scores:

RRF_score(document) = sum over retrieval lists of 1 / (k + rank(document))

Here, rank is the 1-based position of the document in a result list and k is a smoothing constant, commonly set to 60. A document appearing near the top of several lists receives a stronger fused score.

## Example

Suppose dense search ranks passage A first and passage B second, while BM25 ranks B first and passage C second. RRF rewards passage B because both systems independently ranked it highly. The raw score scales do not need to be calibrated against each other.

## Engineering considerations

- Retrieve enough candidates from each first-stage retriever before fusion.
- Deduplicate using stable document or chunk identifiers.
- Keep original per-retriever ranks and scores for debugging.
- Tune candidate depth and final top-k against a labelled evaluation set.
- Consider weighted RRF only when experiments justify different retriever weights.

## Source

Paraphrased learning note informed by:
- Microsoft Learn, *Reciprocal Rank Fusion in RAG information retrieval*: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval
- Google Cloud, *About hybrid search*: https://cloud.google.com/generative-ai-app-builder/docs/hybrid-search
- Docker Docs, *RAG tool — hybrid search*: https://docs.docker.com/ai/docker-agent/tools/rag/
