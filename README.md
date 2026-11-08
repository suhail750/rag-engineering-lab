# RAG Engineering Lab 🧪

A beginner-friendly, open-source RAG app. Visitors can upload PDFs, ask questions about them, and inspect retrieved evidence. It also includes a curated RAG learning pack and retrieval evaluation tools.

Repository: https://github.com/suhail750/rag-engineering-lab

## What visitors can do

- Upload up to 5 PDFs at a time (15 MB per file in this demo).
- Ask questions about the text in their PDFs.
- Inspect retrieved passages and page markers.
- Choose Dense, BM25, or Hybrid retrieval.
- Optionally rerank passages with a cross-encoder.
- Use the included RAG learning pack without uploading a file.
- Run a small retrieval evaluation dataset.

Each visitor's documents and chat are held in that visitor's Streamlit session memory. They are not saved to a shared index file. Session data is temporary and can disappear when a session ends or the app restarts. Do not upload confidential documents.

## LangChain usage

This project uses LangChain in two clear places:

1. Document splitting: raglab/ingestion.py uses Document and RecursiveCharacterTextSplitter.
2. Answer generation: raglab/generation.py builds a simple chain: ChatPromptTemplate | ChatGroq | StrOutputParser.

The embedding and retrieval functions are kept as small Python functions so a beginner can see how vectors, BM25, and hybrid ranking work.

## Run locally

Requires Python 3.10–3.12.

~~~bash
git clone https://github.com/suhail750/rag-engineering-lab.git
cd rag-engineering-lab
python -m venv .venv
~~~

Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
~~~

macOS / Linux:

~~~bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
~~~

Open the local URL printed by Streamlit, usually http://localhost:8501.

## Optional LLM answers

Enter a Groq API key in the sidebar. Without a key, the app displays the most relevant passages instead of pretending that an LLM generated an answer. Never commit API keys to GitHub.

## Included learning pack

The six curated notes cover RAG fundamentals, chunking, dense and BM25 retrieval, hybrid search, reranking, and evaluation. They are paraphrased educational summaries based on public technical documentation and research. Original URLs are preserved in sample_data/sources.json.

To load them, open Knowledge base and click Load free learning pack.

## Evaluation

The included JSON dataset contains seven questions. Load the learning pack, open Evaluation, and upload sample_data/evaluation.json.

Metrics include Hit Rate@k, Recall@k, Precision@k, MRR@k, and nDCG@k. These measure source retrieval, not whether an LLM answer is correct or faithful.

Run tests:

~~~bash
python -m unittest discover -s tests -v
~~~

## Public deployment notes

You can deploy the app using Streamlit Community Cloud by connecting this repository and selecting app.py.

This is an educational demo, not a hardened production service. Session separation prevents one visitor's upload from being placed in another visitor's index, but production deployment still needs server-side rate limits, abuse monitoring, quotas, stronger resource controls, and a durable storage design with explicit tenant isolation. The session index is temporary. Scanned PDFs need OCR. Do not upload private or sensitive files.

## Architecture

~~~text
PDF upload
   ↓
pypdf text extraction + page markers
   ↓
LangChain Document + text splitting
   ↓
Sentence Transformer embeddings
   ↓
Dense / BM25 / Hybrid retrieval
   ↓
Optional cross-encoder reranking
   ↓
LangChain prompt → Groq chat model → answer
   ↓
Evidence passages shown to the visitor
~~~

## Sources

- Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks: https://arxiv.org/abs/2005.11401
- Microsoft Learn, RAG information retrieval phase: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval
- Sentence Transformers, Retrieve & Re-Rank: https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- Sentence Transformers, Information retrieval evaluation: https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html

## License

MIT. See LICENSE.
