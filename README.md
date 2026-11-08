# RAG Engineering Lab 🧪

A beginner-friendly RAG application that answers questions about **Retrieval-Augmented Generation (RAG) engineering** using a curated knowledge base built from public web documentation and research papers.

Repository: https://github.com/suhail750/rag-engineering-lab

## What the app does

- Lets anyone ask questions about RAG engineering.
- Retrieves evidence only from the project's included knowledge base.
- Shows the exact passages used to answer a question.
- Links each learning note to its original public web source.
- Supports Dense, BM25, and Hybrid retrieval.
- Includes optional cross-encoder reranking.
- Includes a built-in retrieval evaluation dataset and metrics.
- Uses LangChain for document splitting and LLM answer generation.

**There is no visitor PDF or data upload feature.** Everyone explores the same curated RAG learning corpus. Chat history is kept in the visitor's Streamlit session.

## Main topic: RAG Engineering

The knowledge base covers:

1. RAG fundamentals and the indexing/query workflows.
2. Document chunking and metadata.
3. Dense retrieval and BM25 lexical retrieval.
4. Hybrid search and Reciprocal Rank Fusion (RRF).
5. Retrieve-then-rerank pipelines.
6. Retrieval and generated-answer evaluation.

The notes are original, beginner-friendly paraphrases. They are not copied full-text web pages. Original source URLs are included in `sample_data/sources.json` and shown in the app.

## LangChain usage

LangChain is used in two understandable places:

1. **Ingestion:** `raglab/ingestion.py` uses `Document` and `RecursiveCharacterTextSplitter` to split notes into chunks.
2. **Generation:** `raglab/generation.py` builds a simple chain: `ChatPromptTemplate | ChatGroq | StrOutputParser`.

The embedding and retrieval functions remain small Python functions so beginners can inspect how the RAG pipeline works.

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

Enter a Groq API key in the sidebar. Without a key, the app returns the most relevant retrieved passages instead of pretending an LLM generated an answer. Never commit API keys to GitHub.

## Evaluation

Open the Evaluation tab and click **Run the built-in evaluation**. The repository includes seven starter questions in `sample_data/evaluation.json`.

Metrics include Hit Rate@k, Recall@k, Precision@k, MRR@k, and nDCG@k. These measure source retrieval, not whether a generated answer is correct or faithful.

Run tests:

~~~bash
python -m unittest discover -s tests -v
~~~

## Public deployment

The app can be deployed using Streamlit Community Cloud by connecting this repository and selecting `app.py`.

This is an educational demo, not a hardened production service. The knowledge base is read-only and shared by all visitors; each visitor has a separate chat history. If you deploy with a Groq API key, store it in the hosting platform's secrets manager rather than in source code.

## Architecture

~~~text
Public web documentation + research papers
                 ↓
       Curated learning notes
                 ↓
   LangChain text splitting / chunking
                 ↓
       Sentence Transformer embeddings
                 ↓
        Dense / BM25 / Hybrid search
                 ↓
      Optional cross-encoder reranking
                 ↓
       LangChain prompt → Groq LLM
                 ↓
     Answer + retrieved evidence + links
~~~

## Sources

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*: https://arxiv.org/abs/2005.11401
- Microsoft Learn, *RAG information retrieval phase*: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval
- Sentence Transformers, *Retrieve & Re-Rank*: https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- Sentence Transformers, *Information retrieval evaluation*: https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html
- Google Cloud, *Hybrid search*: https://cloud.google.com/generative-ai-app-builder/docs/hybrid-search

## License

MIT. See LICENSE.
