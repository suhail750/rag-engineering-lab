# Dense retrieval and BM25 lexical retrieval

A retrieval system selects candidate passages from a corpus for a user query. Dense and lexical retrieval use different signals and can complement each other.

## Dense retrieval

A bi-encoder maps a query and each document passage into vector representations. The system compares the query vector with stored passage vectors, often using cosine similarity or dot product.

Dense retrieval can find passages that express a similar idea using different words. Its quality depends on the embedding model, domain fit, chunking, and similarity configuration. Dense retrieval may struggle with exact identifiers, rare names, product codes, and unusual technical tokens.

## BM25 lexical retrieval

BM25 is a ranking function built on term matching and document statistics. It considers how often query terms occur in a document, how rare those terms are across the corpus, and document length.

BM25 is often useful for exact terminology, error messages, identifiers, names, and queries where literal wording matters. It may miss relevant passages when the query uses different wording from the document.

## Choosing between them

There is no universally best retrieval method. Test both methods on the actual corpus and question distribution. Record retrieval quality and operational cost rather than relying on assumptions.

## Practical diagnostic examples

- Query: "What does RRF stand for?" Exact term matching can be useful.
- Query: "How can search combine meaning-based and keyword-based results?" Dense retrieval may connect the paraphrase to the same concept.
- Query: "Explain error E_CONN_RESET in service ABC-42." Lexical retrieval may be especially valuable for the exact identifier.

## Source

Paraphrased learning note informed by:
- Sentence Transformers, *Retrieve & Re-Rank*: https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- Microsoft Azure Architecture Center, *Information retrieval phase*: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval
