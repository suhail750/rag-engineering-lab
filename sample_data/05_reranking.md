# Retrieve-then-rerank pipelines

A retrieve-then-rerank architecture uses two stages.

## Stage one: candidate retrieval

A fast retriever selects a relatively broad set of candidate passages. Candidate generation can use BM25, dense embeddings, sparse neural retrieval, or hybrid fusion.

The first stage should favor recall: it should retrieve the passages that might answer the question, even if some candidates are imperfect.

## Stage two: reranking

A more expensive model scores each query-passage pair and reorders the candidate set. A cross-encoder jointly processes the query and passage, allowing richer interaction than independently embedding each side.

Reranking can improve precision near the top of the list, but adds latency and compute. It cannot recover a relevant passage that the first stage failed to retrieve.

## Practical workflow

1. Retrieve perhaps 20–100 candidates, depending on corpus size and latency budget.
2. Rerank the candidate passages with a cross-encoder.
3. Keep the best few passages for the generation context.
4. Measure Recall@candidate-depth and nDCG@final-k to understand both stages.
5. Compare latency and cost with and without reranking.

The best candidate depth and final context size are workload-specific. Evaluate on your own representative queries.

## Source

Paraphrased learning note informed by:
- Sentence Transformers, *Retrieve & Re-Rank*: https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- Microsoft Azure Architecture Center, *Language model reranking*: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval
