# Introduction to Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) is a pattern that combines information retrieval with text generation. Instead of relying only on information stored in a language model's parameters, a RAG system retrieves relevant passages from an external knowledge collection and supplies them to the model as context.

## Typical RAG pipeline

A basic RAG pipeline has two phases: ingestion and query time. During ingestion, documents are loaded, cleaned, split into chunks, enriched with metadata, converted into embeddings, and stored in an index. At query time, the user's question is embedded, relevant chunks are retrieved, and the selected context is passed to a language model to produce an answer.

## Why chunking matters

Documents are split into chunks because entire books or large files may be too long or imprecise to retrieve as a single unit. Chunk size controls how much context each passage contains. Overlap preserves information near chunk boundaries, but excessive overlap creates redundant results and increases storage.

## Dense and lexical retrieval

Dense retrieval compares vector representations of a question and document chunks. It can find semantically similar text even when the wording differs. Lexical retrieval such as BM25 uses token matches and term statistics, which can be useful for exact names, identifiers, and technical phrases. Hybrid retrieval combines complementary ranked lists; Reciprocal Rank Fusion is one common rank-based method.

## Reranking

A reranker receives a query and candidate passages and estimates how relevant each passage is to that query. A cross-encoder jointly processes the query and passage, often improving ordering at the cost of additional inference time.

## Evaluation

Retrieval can be evaluated using labelled questions and known relevant source documents. Hit Rate at k measures how often at least one relevant source appears in the top k. Recall at k measures the fraction of relevant sources retrieved. Mean Reciprocal Rank rewards systems that place the first relevant result near the top. These retrieval metrics do not by themselves establish whether a generated answer is correct, complete, or well grounded.
