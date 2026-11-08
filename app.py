from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

from raglab.config import APP_TITLE, DATA_DIR, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from raglab.embeddings import EmbeddingService
from raglab.evaluation import evaluate_retrieval
from raglab.generation import answer_question
from raglab.ingestion import prepare_chunks
from raglab.loaders import load_uploaded_file
from raglab.retrieval import bm25_search, dense_search, hybrid_search, rerank_results
from raglab.storage import load_index, save_index

ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT_DIR / "sample_data"

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


def load_sample_documents() -> list[tuple[str, str, dict]]:
    manifest_path = SAMPLE_DIR / "sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = []
    for item in manifest.get("documents", []):
        file_path = SAMPLE_DIR / item["file"]
        if not file_path.is_file():
            raise FileNotFoundError(f"Sample knowledge file is missing: {item['file']}")
        documents.append((
            item["file"],
            file_path.read_text(encoding="utf-8"),
            {
                "title": item.get("title", item["file"]),
                "source_url": item.get("source_url", ""),
                "publisher": item.get("publisher", ""),
                "source_type": item.get("source_type", "web reference"),
                "curated_pack": manifest.get("pack_name", "RAG starter pack"),
            },
        ))
    return documents


def show_evidence(results: list[dict]) -> None:
    for rank, result in enumerate(results, 1):
        st.markdown(f"**[{rank}] {result.get('source', 'Unknown')}** · score: {result.get('score', 0):.4f}")
        st.write(result.get("text", ""))
        metadata = result.get("metadata", {})
        source_url = metadata.get("source_url")
        if source_url:
            st.markdown(f"[Original source: {metadata.get('publisher') or 'Open source'}]({source_url})")


if "index" not in st.session_state:
    st.session_state.index = load_index()
if "messages" not in st.session_state:
    st.session_state.messages = []

index = st.session_state.index
chunks = index.get("chunks", [])
all_sources = sorted({c.get("source", "unknown") for c in chunks})

with st.sidebar:
    st.title("⚙️ Lab settings")
    st.caption("Settings apply to new queries and ingestion.")
    model_name = st.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = st.slider("Chunk size (characters)", 300, 1800, DEFAULT_CHUNK_SIZE, 50)
    overlap = st.slider("Chunk overlap", 0, 400, DEFAULT_CHUNK_OVERLAP, 25)
    retrieval_mode = st.selectbox("Retrieval strategy", ["Hybrid", "Dense", "BM25"])
    top_k = st.slider("Final results (top-k)", 1, 12, 5)
    use_reranker = st.checkbox("Use cross-encoder reranker", value=False)
    selected_sources = st.multiselect("Filter knowledge sources", all_sources, default=all_sources)
    groq_key = st.text_input("Groq API key (optional)", value=os.getenv("GROQ_API_KEY", ""), type="password")
    groq_model = st.text_input("Groq model", "llama-3.3-70b-versatile")
    st.divider()
    st.caption("Local index is stored in the app's data directory.")
    if st.button("Clear local knowledge base", use_container_width=True):
        st.session_state.index = {"chunks": [], "created_at": None}
        st.session_state.messages = []
        save_index(st.session_state.index)
        st.rerun()

active_chunks = [c for c in chunks if c.get("source", "unknown") in selected_sources]

tab_chat, tab_docs, tab_eval, tab_about = st.tabs(["💬 Chat", "📚 Knowledge base", "📊 Evaluation", "🧭 How it works"])

with tab_docs:
    st.subheader("Add knowledge sources")
    st.write("Start with the curated web-sourced RAG knowledge pack or upload your own files.")

    sample_col, sample_info = st.columns([1, 2])
    with sample_col:
        load_sample = st.button("🌐 Load web knowledge pack", type="primary", use_container_width=True)
    with sample_info:
        st.caption("Six curated learning notes covering RAG, chunking, retrieval, hybrid search, reranking, and evaluation. Original source URLs are preserved.")

    if load_sample:
        try:
            with st.spinner("Loading curated documents and creating embeddings..."):
                documents = load_sample_documents()
                embedder = get_embedder(model_name)
                new_chunks = prepare_chunks(documents, embedder, chunk_size, overlap)
                replace_sources = {name for name, _, _ in documents}
                existing = [c for c in chunks if c.get("source") not in replace_sources]
                st.session_state.index = {"chunks": existing + new_chunks, "created_at": time.time()}
                save_index(st.session_state.index)
            st.success(f"Loaded {len(documents)} web-sourced documents into {len(new_chunks)} searchable chunks.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load the starter knowledge pack: {exc}")

    st.divider()
    uploads = st.file_uploader(
        "Or upload your own documents",
        type=["pdf", "docx", "txt", "md", "csv", "json"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, Markdown, CSV, JSON.",
    )
    if st.button("Process and index uploads", type="secondary", disabled=not uploads):
        try:
            with st.spinner("Extracting text, chunking, and embedding uploads..."):
                documents = []
                for uploaded in uploads:
                    text, metadata = load_uploaded_file(uploaded.name, uploaded.getvalue())
                    documents.append((uploaded.name, text, metadata))
                embedder = get_embedder(model_name)
                new_chunks = prepare_chunks(documents, embedder, chunk_size, overlap)
                replace_sources = {name for name, _, _ in documents}
                existing = [c for c in chunks if c.get("source") not in replace_sources]
                st.session_state.index = {"chunks": existing + new_chunks, "created_at": time.time()}
                save_index(st.session_state.index)
            st.success(f"Indexed {len(new_chunks)} chunks from {len(documents)} uploaded file(s).")
            st.rerun()
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
            st.info("For scanned PDFs, OCR is needed before text can be extracted.")

    if chunks:
        st.caption(f"{len(chunks)} chunks indexed across {len(all_sources)} source(s). The source filter is in the sidebar.")
        for source in all_sources:
            source_chunks = [c for c in chunks if c.get("source") == source]
            with st.expander(f"📄 {source} · {len(source_chunks)} chunks"):
                metadata = source_chunks[0].get("metadata", {})
                st.write(metadata)
                st.text(source_chunks[0].get("text", "")[:1200])
                if metadata.get("source_url"):
                    st.markdown(f"[Open original source]({metadata['source_url']})")
    else:
        st.info("Your knowledge base is empty. Load the web knowledge pack or upload documents.")

with tab_chat:
    st.subheader("Ask your knowledge base")
    st.caption("Answers use retrieved passages. Inspect the evidence and open original source links.")
    if not active_chunks:
        st.info("Load the starter knowledge pack or upload documents in the Knowledge base tab. Check your source filter too.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("🔎 Retrieved evidence"):
                    show_evidence(message["sources"])

    question = st.chat_input("Ask a question about your documents...", disabled=not active_chunks)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                start = time.perf_counter()
                embedder = get_embedder(model_name)
                query_vector = embedder.encode([question])[0]
                if retrieval_mode == "Dense":
                    results = dense_search(query_vector, active_chunks, top_k=top_k)
                elif retrieval_mode == "BM25":
                    results = bm25_search(question, active_chunks, top_k=top_k)
                else:
                    results = hybrid_search(question, query_vector, active_chunks, top_k=top_k)
                if use_reranker and results:
                    results = rerank_results(question, results, top_k=top_k)
                response = answer_question(question, results, api_key=groq_key, model=groq_model)
                elapsed_ms = (time.perf_counter() - start) * 1000
                st.markdown(response["answer"])
                st.caption(f"{retrieval_mode} retrieval · {elapsed_ms:.0f} ms · {len(results)} evidence chunks · {response['mode']}")
                with st.expander("🔎 Inspect retrieved evidence", expanded=True):
                    show_evidence(results)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": [{**r, "rank": i} for i, r in enumerate(results, 1)],
                })
            except Exception as exc:
                st.error(f"Could not answer this question: {exc}")
                st.info("Check the model download, API key, and selected knowledge sources, then retry.")

with tab_eval:
    st.subheader("Retrieval evaluation")
    st.write("Use labelled questions to compare retrieval quality. The included dataset is aligned with the starter knowledge pack.")
    example_path = SAMPLE_DIR / "evaluation.json"
    if example_path.is_file():
        st.download_button("Download starter evaluation set", example_path.read_text(encoding="utf-8"), "rag_evaluation.json", "application/json")
    eval_file = st.file_uploader("Evaluation dataset (JSON)", type=["json"], key="eval_upload")
    if eval_file and st.button("Run retrieval evaluation", type="primary", disabled=not active_chunks):
        try:
            dataset = json.loads(eval_file.getvalue().decode("utf-8"))
            if not isinstance(dataset, list) or not dataset:
                raise ValueError("The evaluation file must contain a non-empty JSON list.")
            rows = []
            embedder = get_embedder(model_name)
            for item in dataset:
                if not isinstance(item, dict) or not isinstance(item.get("question"), str) or not isinstance(item.get("relevant_sources"), list):
                    raise ValueError("Each test case needs a question string and relevant_sources list.")
                q = item["question"]
                qvec = embedder.encode([q])[0]
                if retrieval_mode == "Dense":
                    found = dense_search(qvec, active_chunks, top_k=top_k)
                elif retrieval_mode == "BM25":
                    found = bm25_search(q, active_chunks, top_k=top_k)
                else:
                    found = hybrid_search(q, qvec, active_chunks, top_k=top_k)
                rows.append({"relevant_sources": item["relevant_sources"], "retrieved_sources": [r.get("source") for r in found]})
            metrics = evaluate_retrieval(rows, k=top_k)
            cols = st.columns(5)
            cols[0].metric(f"Hit Rate@{top_k}", f"{metrics['hit_rate']:.1%}")
            cols[1].metric(f"Recall@{top_k}", f"{metrics['recall']:.1%}")
            cols[2].metric(f"Precision@{top_k}", f"{metrics['precision']:.1%}")
            cols[3].metric(f"MRR@{top_k}", f"{metrics['mrr']:.3f}")
            cols[4].metric(f"nDCG@{top_k}", f"{metrics['ndcg']:.3f}")
            st.caption(f"Evaluated {len(rows)} questions. These metrics measure source retrieval, not answer correctness.")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

with tab_about:
    st.subheader("The RAG pipeline")
    st.markdown("""
    1. **Load:** extract text and metadata from uploaded files or the curated web starter pack.
    2. **Chunk:** split text into overlapping passages.
    3. **Embed:** map passages and questions into vectors.
    4. **Retrieve:** dense search finds semantic matches; BM25 finds lexical matches; hybrid search fuses ranked lists with Reciprocal Rank Fusion.
    5. **Rerank (optional):** a cross-encoder scores query-passage pairs for relevance.
    6. **Generate:** the LLM receives selected context and is instructed to cite evidence. Without an API key, an extractive fallback is used.
    7. **Evaluate:** labelled questions measure Hit Rate, Recall@k, Precision@k, MRR, and nDCG.
    """)
    st.warning("This is an educational single-user app. Before public multi-user use, add authentication, per-user indexes, quotas, durable storage, rate limiting, and abuse protection. Never upload private or confidential documents to an open public deployment.")
