# RAG evaluation, metrics, and production

A RAG system has two connected quality problems: did search find the right evidence, and did the LLM use that evidence correctly? Evaluate both.

## Retrieval metrics
- **Precision@k:** fraction of the top-k results that are relevant.
- **Recall@k:** fraction of all known relevant results found in the top-k.
- **Hit Rate@k:** proportion of questions where at least one relevant result appears in top-k.
- **MRR:** rewards finding the first relevant result near the top.
- **nDCG:** rewards highly relevant results appearing earlier and supports graded relevance.

Example: if 4 of the first 5 passages are relevant, Precision@5 = 4/5 = 0.8. If there are 8 relevant passages in the corpus and 4 were retrieved, Recall@5 = 4/8 = 0.5.

## Answer metrics and checks
Retrieval scores do not prove that the final answer is correct. Evaluate:
- **Correctness:** does the answer match reliable ground truth?
- **Faithfulness / groundedness:** are claims supported by retrieved context?
- **Answer relevance:** does it answer the question?
- **Completeness:** does it cover the important parts?
- **Citation accuracy:** do citations support the claims?
- **Safety:** does it avoid unsafe or unsupported advice?

## Three evaluation approaches
- **Deterministic:** exact match, token overlap, precision, recall, MRR, nDCG, schema checks, latency, and cost. Repeatable, but may not understand meaning.
- **Model-based:** an evaluator model scores faithfulness, relevance, or correctness using a rubric. Scalable, but can be inconsistent or biased.
- **Human-based:** people judge usefulness, correctness, clarity, and safety. Context-aware, but takes time and needs clear rubrics.

Use a combination; no single metric fully describes RAG quality.

## Production checklist
- Monitor ingestion failures, empty retrieval, latency, token usage, and model errors.
- Track document freshness and remove outdated or duplicate content.
- Enforce user permissions before returning content.
- Record source IDs and retrieval scores without exposing sensitive data unnecessarily.
- Test prompt injection in documents and user questions.
- Re-run evaluation after changing chunking, embeddings, top-k, reranking, prompts, or models.

## Improvement loop
1. Save realistic user questions.
2. Inspect retrieved passages when answers fail.
3. Identify whether the problem is ingestion, chunking, embeddings, retrieval, reranking, or generation.
4. Change one component at a time.
5. Re-run the same evaluation set.
6. Compare quality, latency, and cost.

Sources:
- https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval
- https://arxiv.org/abs/2005.11401
