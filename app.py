from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

from raglab.chunking import chunk_text
from raglab.config import APP_TITLE, DATA_DIR, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from raglab.embeddings import EmbeddingService
from raglab.evaluation import evaluate_retrieval
from raglab.generation import answer_question
from raglab.loaders import load_uploaded_file
from raglab.retrieval import bm25_search, dense_search, hybrid_search, rerank_results
from raglab.storage import load_index, save_index

st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.7rem; max-width: 1440px;}
.hero {padding: 1.5rem 1.7rem; border: 1px solid rgba(128,128,128,.25); border-radius: 16px; margin-bottom: 1rem;}
.hero h1 {margin: 0 0 .35rem 0; font-size: 2.1rem;}
.muted {color: #8b929e;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.2); padding: 12px; border-radius: 12px;}
</style>
<div class="hero">
  <div class="muted">OPEN-SOURCE RAG WORKBENCH</div>
  <h1>🧪 RAG Engineering Lab</h1>
  <div class="muted">Build, inspect, compare, and evaluate retrieval-augmented generation systems.</div>
</div>
""", unsafe_allow_html=True)

DATA_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def get_embedder(model_name: str) -> EmbeddingService:
    return EmbeddingService(model_name)

if "index" not in st.session_state:
    st.session_state.index = load_index()
if "messages" not in st.session_state:
    st.session_state.messages = []

index = st.session_state.index
chunks = index.get("chunks", [])

with st.sidebar:
    st.title("⚙️ Lab settings")
    st.caption("Settings apply to new queries and ingestion.")
    model_name = st.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = st.slider("Chunk size (characters)", 300, 1800, DEFAULT_CHUNK_SIZE, 50)
    overlap = st.slider("Chunk overlap", 0, 400, DEFAULT_CHUNK_OVERLAP, 25)
    retrieval_mode = st.selectbox("Retrieval strategy", ["Hybrid", "Dense", "BM25"])
    top_k = st.slider("Final results (top-k)", 1, 12, 5)
    use_reranker = st.checkbox("Use cross-encoder reranker", value=False)
    groq_key = st.text_input("Groq API key (optional)", value=os.getenv("GROQ_API_KEY", ""), type="password")
    groq_model = st.text_input("Groq model", "llama-3.3-70b-versatile")
    st.divider()
    st.caption("Local index stored in the app's data directory.")
    if st.button("Clear local knowledge base", use_container_width=True):
        st.session_state.index = {"chunks": [], "created_at": None}
        st.session_state.messages = []
        save_index(st.session_state.index)
        st.rerun()

tab_chat, tab_docs, tab_eval, tab_about = st.tabs(["💬 Chat", "📚 Knowledge base", "📊 Evaluation", "🧭 How it works"])

with tab_docs:
    st.subheader("Add knowledge sources")
    st.write("Upload files to extract text, split it into chunks, and build the local searchable index.")
    uploads = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt", "md", "csv", "json"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, Markdown, CSV, JSON.",
    )
    col_a, col_b = st.columns([1, 3])
    with col_a:
        ingest_clicked = st.button("Process and index", type="primary", disabled=not uploads)
    with col_b:
        st.caption("Large files and first-time embedding downloads may take a while.")
    if ingest_clicked and uploads:
        try:
            embedder = get_embedder(model_name)
            new_chunks = []
            progress = st.progress(0, text="Preparing documents...")
            for file_num, uploaded in enumerate(uploads):
                text, metadata = load_uploaded_file(uploaded.name, uploaded.getvalue())
                pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                for chunk_num, piece in enumerate(pieces):
                    new_chunks.append({
                        "text": piece,
                        "source": uploaded.name,
                        "chunk_id": f"{uploaded.name}:{chunk_num}",
                        "metadata": {**metadata, "chunk_index": chunk_num},
                    })
                progress.progress((file_num + 1) / len(uploads), text=f"Extracted {uploaded.name}")
            if not new_chunks:
                st.warning("No readable text was found in the selected files.")
            else:
                vectors = embedder.encode([item["text"] for item in new_chunks])
                for item, vector in zip(new_chunks, vectors):
                    item["embedding"] = vector.tolist()
                existing = [c for c in chunks if c.get("source") not in {u.name for u in uploads}]
                st.session_state.index = {"chunks": existing + new_chunks, "created_at": time.time()}
                save_index(st.session_state.index)
                st.success(f"Indexed {len(new_chunks)} chunks from {len(uploads)} file(s).")
                st.rerun()
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
            st.info("Check the file format and try again. For PDFs, make sure text is selectable (scanned PDFs need OCR).")

if chunks:
    st.caption(f"{len(chunks)} chunks currently indexed across {len(set(c.get('source', 'unknown') for c in chunks))} source(s).")
    sources = sorted(set(c.get("source", "unknown") for c in chunks))
    for source in sources:
        source_chunks = [c for c in chunks if c.get("source") == source]
        with st.expander(f"📄 {source} · {len(source_chunks)} chunks"):
            st.write(source_chunks[0].get("metadata", {}))
            st.text(source_chunks[0].get("text", "")[:1200])
else:
    st.info("Your knowledge base is empty. Add documents in this tab to get started.")

with tab_chat:
    st.subheader("Ask your knowledge base")
    st.caption("Answers are grounded in retrieved passages. Open the evidence panel to inspect what the system used.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("🔎 Retrieved evidence"):
                    for result in message["sources"]:
                        st.markdown(f"**[{result['rank']}] {result.get('source', 'Unknown')}** · score: {result.get('score', 0):.4f}")
                        st.write(result.get("text", ""))
    question = st.chat_input("Ask a question about your documents...", disabled=not chunks)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                start = time.perf_counter()
                embedder = get_embedder(model_name)
                query_vector = embedder.encode([question])[0].tolist()
                if retrieval_mode == "Dense":
                    results = dense_search(query_vector, chunks, top_k=top_k)
                elif retrieval_mode == "BM25":
                    results = bm25_search(question, chunks, top_k=top_k)
                else:
                    results = hybrid_search(question, query_vector, chunks, top_k=top_k)
                if use_reranker and results:
                    results = rerank_results(question, results, top_k=top_k)
                response = answer_question(question, results, api_key=groq_key, model=groq_model)
                elapsed_ms = (time.perf_counter() - start) * 1000
                st.markdown(response["answer"])
                st.caption(f"{retrieval_mode} retrieval · {elapsed_ms:.0f} ms · {len(results)} evidence chunk(s) · {response['mode']}")
                with st.expander("🔎 Inspect retrieved evidence"):
                    for rank, result in enumerate(results, 1):
                        st.markdown(f"**[{rank}] {result.get('source', 'Unknown')}** · score: {result.get('score', 0):.4f}")
                        st.write(result.get("text", ""))
                st.session_state.messages.append({"role": "assistant", "content": response["answer"], "sources": [{**r, "rank": i} for i, r in enumerate(results, 1)]})
            except Exception as exc:
                st.error(f"Could not answer this question: {exc}")
                st.info("If this is the first run, wait for the embedding model to finish downloading and retry.")

with tab_eval:
    st.subheader("Retrieval evaluation")
    st.write("Measure whether relevant source documents appear in the retrieved top-k results. Upload a JSON test set.")
    example = [{"question": "What is the main idea?", "relevant_sources": ["example.pdf"]}]
    st.download_button("Download example evaluation JSON", json.dumps(example, indent=2), "evaluation_example.json", "application/json")
    eval_file = st.file_uploader("Evaluation dataset (JSON)", type=["json"], key="eval_upload")
    if eval_file and st.button("Run retrieval evaluation", type="primary"):
        try:
            dataset = json.loads(eval_file.getvalue().decode("utf-8"))
            if not isinstance(dataset, list):
                raise ValueError("The evaluation file must contain a JSON list.")
            embedder = get_embedder(model_name)
            rows = []
            for item in dataset:
                q = str(item["question"])
                relevant = item["relevant_sources"]
                qvec = embedder.encode([q])[0].tolist()
                if retrieval_mode == "Dense":
                    found = dense_search(qvec, chunks, top_k=top_k)
                elif retrieval_mode == "BM25":
                    found = bm25_search(q, chunks, top_k=top_k)
                else:
                    found = hybrid_search(q, qvec, chunks, top_k=top_k)
                rows.append({"relevant_sources": relevant, "retrieved_sources": [r.get("source") for r in found]})
            metrics = evaluate_retrieval(rows, k=top_k)
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Hit Rate@{top_k}", f"{metrics['hit_rate']:.1%}")
            c2.metric(f"Recall@{top_k}", f"{metrics['recall']:.1%}")
            c3.metric(f"MRR@{top_k}", f"{metrics['mrr']:.3f}")
            st.caption(f"Evaluated {len(rows)} question(s). These are retrieval metrics, not answer-quality scores.")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

with tab_about:
    st.subheader("The RAG pipeline")
    st.markdown("""
    1. **Load:** extract text and metadata from uploaded files.
    2. **Chunk:** split text into overlapping passages.
    3. **Embed:** map passages and questions into vectors.
    4. **Retrieve:** dense search finds semantic matches; BM25 finds lexical matches; hybrid search fuses ranked lists with Reciprocal Rank Fusion.
    5. **Rerank (optional):** a cross-encoder scores query-passage pairs for relevance.
    6. **Generate:** the LLM receives only the selected context and is instructed to cite evidence. Without an API key, an extractive fallback is used.
    7. **Evaluate:** labelled source references let you calculate Hit Rate, Recall@k, and Mean Reciprocal Rank.
    """)
    st.warning("This is an educational single-user starter. A public multi-user deployment needs authentication, per-user indexes, quotas, durable storage, rate limiting, and abuse protection before handling real user data.")
