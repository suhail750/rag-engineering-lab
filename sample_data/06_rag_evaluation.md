# Evaluating retrieval and generated answers

A RAG system has multiple components, so one score cannot describe its full quality. Evaluate retrieval, generation, safety, and operational performance separately.

## Retrieval metrics

- Hit Rate@k: fraction of queries for which at least one relevant result appears in the top k.
- Recall@k: fraction of all labelled relevant documents retrieved in the top k, averaged over queries.
- Precision@k: fraction of the top k retrieved results that are relevant.
- Mean Reciprocal Rank (MRR): average reciprocal rank of the first relevant result.
- nDCG@k: ranking metric that rewards placing highly relevant results near the top and can account for graded relevance.

Metrics require a labelled query set with relevance judgements. Source-level labels are a useful starting point, but chunk-level relevance labels provide a more precise view of passage retrieval.

## Generation quality

Evaluate generated answers separately for:
- Correctness against a trusted reference answer.
- Faithfulness to the retrieved evidence.
- Completeness of the answer.
- Citation correctness: whether each citation supports the associated claim.
- Appropriate abstention when evidence is missing.

Human review is important for high-impact use cases. Automated model-based judges can help scale evaluation, but judge outputs should be calibrated against human-labelled examples.

## System performance

Track latency by stage, model/provider, token usage, embedding cost, retrieval depth, errors, and cache hit rate. A quality improvement may not be worthwhile if it makes the system too slow or expensive for the intended workload.

## Experimental discipline

Keep a fixed evaluation dataset. Change one major variable at a time where possible. Record configuration, dataset version, and results so experiments can be reproduced. Avoid reporting a metric without explaining the dataset and evaluation method.

## Source

Paraphrased learning note informed by:
- Sentence Transformers, *InformationRetrievalEvaluator*: https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html
- Sentence Transformers, *ReciprocalRankFusionEvaluator*: https://www.sbert.net/docs/package_reference/sparse_encoder/evaluation.html
