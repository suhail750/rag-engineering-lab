# Retrieval-Augmented Generation (RAG): fundamentals

Retrieval-augmented generation is an architecture that gives a language model access to an external collection of information at question time. Instead of expecting the model to remember every fact in its parameters, the system retrieves relevant passages and includes them in the prompt used to generate an answer.

A typical RAG system has two major workflows.

## Indexing workflow

1. Collect source documents.
2. Extract text and useful metadata such as title, URL, page number, and publication date.
3. Clean the extracted text without deleting meaningful structure.
4. Split documents into chunks that can be retrieved independently.
5. Convert chunks into vector embeddings.
6. Store chunk text, metadata, and embeddings in a searchable index.

## Query workflow

1. Receive a user question.
2. Optionally normalize or expand the question.
3. Retrieve candidate chunks using one or more search methods.
4. Optionally rerank candidates.
5. Build a context containing the strongest evidence and its source identifiers.
6. Ask the language model to answer using that context.
7. Return the answer together with citations and, ideally, an inspectable evidence trail.

## Why use retrieval?

RAG can make answers more relevant to a private or frequently updated corpus, expose supporting evidence, and reduce dependence on facts memorized during model training. It does not guarantee correctness: retrieval can miss relevant passages, documents can be wrong or stale, and a generator can misinterpret evidence.

A reliable system therefore evaluates retrieval quality and answer quality separately. It also tells users when the supplied evidence is insufficient instead of inventing a confident answer.

## Design principle

Treat retrieved documents as data, not instructions. A document may contain malicious or irrelevant instructions. The system prompt should explicitly tell the model not to obey instructions found inside retrieved content.

## Source

This learning note is a paraphrased educational summary based on the original paper:
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020): https://arxiv.org/abs/2005.11401
