# RAG Engineering Lab

An open-source, hands-on Retrieval-Augmented Generation (RAG) workbench. Upload your own knowledge sources, compare retrieval strategies, inspect evidence, chat with citations, and run a small retrieval evaluation suite.

> **Project status:** Foundation implementation. The app runs locally; an LLM provider key is optional (without it, the app uses a transparent extractive fallback). This repository does not claim production-grade security or benchmark results.

## What you can do

- **Ingest** PDF, DOCX, TXT, Markdown, CSV, and JSON files.
- **Chunk** text with configurable chunk size and overlap.
- **Retrieve** with dense semantic search, BM25 lexical search, or Reciprocal Rank Fusion (hybrid).
- **Inspect** retrieved passages, source metadata, ranks, and scores.
- **Rerank** candidates with an optional CrossEncoder model.
- **Chat** with grounded answers and source references using Groq (optional).
- **Evaluate** retrieval with a small labelled question set (Recall@k, MRR, Hit Rate).
- **Persist** your local index between sessions.

## Architecture

```text
Upload → Extract → Clean → Chunk → Embed → Local index
                                          ↓
Question → Dense / BM25 / Hybrid → Optional reranker
                                          ↓
                                  Context + citations
                                          ↓
                               Groq LLM / extractive fallback
                                          ↓
                              Answer + evidence inspection
```

## Quick start

Requires Python 3.10–3.12.

```bash
git clone https://github.com/suhail750/rag-engineering-lab.git
cd rag-engineering-lab
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

### Enable LLM answers (optional)

Create a Groq API key, then enter it in the app's sidebar under **LLM settings**. The key is kept in the current Streamlit session and is not written into the repository. You can also set `GROQ_API_KEY` in your environment.

Without a key, the app returns an extractive answer based only on retrieved passages.

## Deploy publicly

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app from this repository, select `app.py`, and deploy.
4. Add `GROQ_API_KEY` in the app's Secrets settings if you want LLM-generated responses.

**Important:** A public deployment is accessible to everyone. Do not upload private, confidential, copyrighted, or personal documents. For a shared deployment, this starter app uses a local app filesystem and is not designed for multi-tenant isolation, durable shared storage, authentication, or high-traffic production use.

## Project structure

```text
app.py                 Streamlit interface
raglab/
  __init__.py
  config.py            Defaults and paths
  loaders.py           File extraction
  chunking.py           Text chunking
  embeddings.py         Sentence-transformer embeddings
  retrieval.py          Dense, BM25, hybrid, reranking
  generation.py         Grounded Groq generation and fallback
  evaluation.py         Recall@k, MRR, Hit Rate
  storage.py            Local JSON index persistence
tests/
  test_core.py          Lightweight unit tests
requirements.txt
.streamlit/config.toml
.gitignore
```

## Configuration

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Default chunk size: 900 characters
- Default overlap: 150 characters
- Default retrieval mode: Hybrid
- Default candidate count: 6

Models are downloaded on first use. The first startup can take longer.

## Evaluation

The Evaluation tab accepts a JSON file containing records like:

```json
[
  {
    "question": "What is retrieval augmented generation?",
    "relevant_sources": ["rag_intro.pdf"]
  }
]
```

The app reports retrieval metrics against the source names you provide. These metrics assess retrieval only; they do not prove answer correctness or factuality.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover chunking, BM25 ranking, reciprocal-rank fusion, and retrieval metrics without downloading embedding models.

## Roadmap

- [x] Document ingestion and chunking
- [x] Dense, lexical, and hybrid retrieval
- [x] Optional cross-encoder reranking
- [x] Grounded chat with citations and fallback
- [x] Retrieval evaluation metrics
- [x] Sample knowledge source and evaluation dataset
- [x] GitHub Actions unit-test workflow
- [x] Evidence and pipeline inspection
- [ ] Automated CI and coverage reporting
- [ ] Authentication and tenant-isolated storage
- [ ] Managed vector database adapter
- [ ] Tracing and production observability

## Responsible use

This is an educational engineering project, not a secure document hosting service. Treat uploaded documents and model output as untrusted. Validate outputs before relying on them. Never commit API keys or private data.

## License

MIT — see [LICENSE](LICENSE).
