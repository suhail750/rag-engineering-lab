# RAG Engineering Lab 🧪

An open-source RAG workbench for learning and experimenting with document ingestion, retrieval, reranking, grounded generation, and evaluation.

**Repository:** https://github.com/suhail750/rag-engineering-lab

## Try the web-sourced starter knowledge pack

The app includes six curated educational notes based on public technical documentation and research:
1. RAG fundamentals
2. Chunking and metadata
3. Dense retrieval and BM25
4. Hybrid search and Reciprocal Rank Fusion
5. Retrieve-then-rerank
6. RAG evaluation

The notes are original paraphrased learning material, not copied full-text web pages. Original source URLs and publishers are preserved in `sample_data/sources.json` and displayed alongside retrieved evidence.

**To load them:** open the app → **Knowledge base** → **Load web knowledge pack**. The first run downloads the embedding model and can take a few minutes.

A matching seven-question evaluation set is available at `sample_data/evaluation.json` and in the app's Evaluation tab.

## Features

- PDF, DOCX, TXT, Markdown, CSV, and JSON ingestion.
- Configurable chunk size and overlap.
- Sentence Transformer dense embeddings.
- BM25 lexical retrieval.
- Hybrid retrieval using Reciprocal Rank Fusion.
- Optional CrossEncoder reranking.
- Source-aware evidence inspection and source URL links.
- Optional Groq-powered grounded generation; extractive fallback without an API key.
- Source-level Hit Rate@k, Recall@k, Precision@k, MRR@k, and nDCG@k.
- Local JSON index persistence and source filtering.
- GitHub Actions unit tests.

## Architecture

```text
Documents / curated web notes
          ↓
     Text extraction
          ↓
   Chunking + metadata
          ↓
     Embeddings → Local index
          ↓
Question → Dense search ─┐
          BM25 search ────┼→ RRF → optional CrossEncoder
                         ┘
                              ↓
                      Evidence + citations
                              ↓
                    Groq LLM / fallback
                              ↓
                      Answer + sources
                              ↓
                  Retrieval evaluation
```

## Run locally

Requires Python 3.10–3.12.

```bash
git clone https://github.com/suhail750/rag-engineering-lab.git
cd rag-engineering-lab
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

### Optional LLM answers

Add a Groq API key in the app sidebar or set the `GROQ_API_KEY` environment variable. Without a key, the app returns relevant retrieved passages rather than pretending that an LLM generated an answer.

Never commit a real API key. See `.env.example`.

## Evaluation

The included evaluation set is aligned to the sample knowledge pack. Load the pack, open the Evaluation tab, and upload `sample_data/evaluation.json` (or download it from the app).

Metrics:
- **Hit Rate@k:** percentage of questions with at least one relevant source in the top k.
- **Recall@k:** fraction of labelled relevant sources retrieved in the top k.
- **Precision@k:** fraction of the top k slots occupied by relevant sources.
- **MRR@k:** reciprocal rank of the first relevant source, averaged across questions.
- **nDCG@k:** ranking quality that rewards relevant sources appearing earlier.

These are source-level retrieval metrics. They do not prove that generated answers are correct or faithful. Build and label a representative evaluation set before making performance claims.

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

## Deploy

The app can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/):
1. Connect this GitHub repository.
2. Select `app.py` as the entry point.
3. Add `GROQ_API_KEY` in the deployment secrets if LLM answers are desired.

**Public deployment warning:** this current starter is a single shared app with local filesystem storage. It is not multi-tenant or safe for confidential user uploads. Before opening it to public uploads, add authentication, per-user storage and indexes, quotas, file limits, rate limiting, durable storage, and abuse protections. Do not upload private or sensitive documents.

## Repository layout

```text
app.py
raglab/
  chunking.py
  config.py
  embeddings.py
  evaluation.py
  generation.py
  ingestion.py
  loaders.py
  retrieval.py
  storage.py
sample_data/
  01_rag_fundamentals.md
  02_chunking_and_metadata.md
  03_dense_and_bm25_retrieval.md
  04_hybrid_search_rrf.md
  05_reranking.md
  06_rag_evaluation.md
  sources.json
  evaluation.json
tests/
  test_core.py
.github/workflows/tests.yml
requirements.txt
```

## Roadmap

- [x] Multi-format ingestion and configurable chunking
- [x] Dense, BM25, and hybrid retrieval
- [x] Optional cross-encoder reranking
- [x] Grounded Groq generation and transparent fallback
- [x] Source citations and evidence inspection
- [x] Curated web-sourced knowledge pack
- [x] Hit Rate, Recall, Precision, MRR, and nDCG
- [x] Unit tests and GitHub Actions
- [ ] Query rewriting, multi-query, and HyDE
- [ ] Metadata filters beyond source selection
- [ ] Parent-child and contextual retrieval
- [ ] Answer-quality and faithfulness evaluation
- [ ] Tracing, caching, and cost analytics
- [ ] Production auth, tenant isolation, and managed vector storage

## Sources and attribution

See [sample_data/sources.json](sample_data/sources.json) for the document-to-source mapping. The knowledge notes are paraphrased educational summaries. Please consult and cite the original publishers for authoritative details.

Key references:
- Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
- Microsoft Learn, [RAG information retrieval phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval).
- Sentence Transformers, [Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html).
- Sentence Transformers, [Information retrieval evaluation](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html).
- Google Cloud, [Hybrid search](https://cloud.google.com/generative-ai-app-builder/docs/hybrid-search).

## License

MIT. See [LICENSE](LICENSE).
