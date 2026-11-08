# RAG fundamentals and end-to-end architecture

Retrieval-Augmented Generation (RAG) gives a large language model (LLM) access to external information when it answers. Instead of relying only on what it learned during training, the application searches a knowledge base, retrieves relevant passages, and supplies them with the question as context.

Think of an open-book exam: the LLM is the student, the knowledge base is the book, retrieval finds the relevant pages, and generation writes the answer using those pages.

## Why use RAG?
- **Fresh information:** update documents without retraining the LLM.
- **Private or domain-specific data:** answer from manuals, research, or company documents.
- **Traceability:** show which passages support an answer.
- **Control:** choose which data the model can access.

RAG can reduce unsupported answers, but it does not eliminate hallucinations. Bad retrieval or misleading context can still produce wrong answers.

## The RAG pipeline
1. Collect data from PDFs, websites, Markdown, databases, or other sources.
2. Parse and clean the content while preserving useful structure.
3. Chunk long documents into smaller passages.
4. Embed each passage into a numerical vector.
5. Index vectors, text, and metadata for search.
6. Embed the user's question and retrieve relevant passages.
7. Augment the prompt with the question and retrieved context.
8. Generate an answer grounded in that context.
9. Evaluate retrieval and answer quality.

Ingestion (collect, parse, chunk, embed, index) usually happens before a user asks a question. At query time, the system searches, optionally reranks, and sends selected passages to the LLM.

## RAG versus fine-tuning
RAG supplies external information at inference time. Fine-tuning updates model parameters using training examples. RAG is useful when facts change or answers need document evidence. Fine-tuning is useful for changing model behaviour, style, or task patterns. They can be combined.

## Limitations
RAG only uses what its pipeline can find. It can fail because of missing documents, poor chunking, weak embeddings, irrelevant retrieval, or noisy prompts.

## Public sources
- Microsoft Learn: https://learn.microsoft.com/azure/ai-studio/concepts/retrieval-augmented-generation
- Original RAG research paper: https://arxiv.org/abs/2005.11401
- Pinecone RAG guide: https://www.pinecone.io/learn/retrieval-augmented-generation/
