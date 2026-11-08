# Document chunking and metadata

Chunking divides long documents into smaller passages that can be indexed and retrieved independently. The goal is not simply to make every chunk the same length. The goal is to create evidence units that are large enough to preserve meaning and small enough to retrieve precisely.

## Chunk size trade-offs

- Very small chunks can lose context, definitions, and relationships between sentences.
- Very large chunks can contain unrelated topics and consume more of the model's context window.
- Overlap can preserve information across boundaries, but excessive overlap creates duplicated evidence and increases storage and retrieval noise.

## Boundary-aware splitting

When possible, split on meaningful boundaries such as headings, paragraphs, sentences, or sections. For code, preserve complete functions or classes where possible. For tables, preserve headers with their rows so values remain interpretable.

A practical starting point is to experiment with several chunk sizes and overlap settings rather than assuming one configuration is universally best. Evaluate on representative questions from the intended domain.

## Metadata

Store useful metadata with every chunk:
- Stable document ID and chunk ID.
- Original filename or document title.
- Source URL or citation.
- Page number or section heading when available.
- Document type and ingestion timestamp.
- Access-control labels when the corpus has different permission levels.

Metadata supports source citations, filtering, debugging, freshness policies, and access control. Do not rely on a filename alone as a globally unique identifier.

## Evaluate chunking

Create a fixed set of representative questions and relevant source passages. Compare chunk configurations using retrieval metrics such as Recall@k, MRR, and nDCG, then manually inspect whether retrieved passages contain enough context to support an answer.

## Source

Paraphrased learning note informed by:
- Microsoft Azure Architecture Center, *Develop a RAG solution — Information retrieval phase*: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval
- Sentence Transformers, *Retrieve & Re-Rank*: https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
