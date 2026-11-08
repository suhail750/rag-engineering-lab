import json, os, time
from pathlib import Path
import streamlit as st
from raglab.config import APP_TITLE, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from raglab.embeddings import EmbeddingService
from raglab.evaluation import evaluate_retrieval
from raglab.generation import answer_question
from raglab.ingestion import prepare_chunks
from raglab.loaders import load_uploaded_file
from raglab.retrieval import bm25_search, dense_search, hybrid_search, rerank_results

ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT_DIR / "sample_data"
MAX_FILES = 5
MAX_MB = 15

st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1380px;}
.hero {padding: 2rem; border: 1px solid rgba(139,124,255,.35); border-radius: 22px; background: linear-gradient(120deg, rgba(139,124,255,.16), rgba(35,42,61,.35));}
.hero h1 {font-size: 2.6rem; margin-bottom: .5rem;}
.hero p {color: #C7CBD6; font-size: 1.05rem;}
</style>
<div class="hero"><h1>🧪 RAG Engineering Lab</h1>
<p>Upload your PDFs, ask questions in plain English, and inspect the evidence behind every answer.</p></div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder(name):
    return EmbeddingService(name)

def sample_documents():
    manifest = json.loads((SAMPLE_DIR / "sources.json").read_text(encoding="utf-8"))
    result = []
    for item in manifest.get("documents", []):
        path = SAMPLE_DIR / item["file"]
        result.append((item["file"], path.read_text(encoding="utf-8"), {
            "title": item.get("title", item["file"]),
            "source_url": item.get("source_url", ""),
            "publisher": item.get("publisher", ""),
            "source_type": item.get("source_type", "web reference"),
        }))
    return result

def show_evidence(results):
    for rank, result in enumerate(results, 1):
        st.markdown(f"**[{rank}] {result.get('source', 'Unknown')}** · score: {result.get('score', 0):.4f}")
        st.write(result.get("text", ""))
        metadata = result.get("metadata", {})
        if metadata.get("source_url"):
            st.markdown(f"[Original source: {metadata.get('publisher') or 'Open source'}]({metadata['source_url']})")

# Session state is isolated per visitor. Uploaded files are never saved to a shared index.
if "index" not in st.session_state:
    st.session_state.index = {"chunks": [], "created_at": None}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "notice" not in st.session_state:
    st.session_state.notice = ""

chunks = st.session_state.index["chunks"]
sources = sorted({item.get("source", "unknown") for item in chunks})
with st.sidebar:
    st.title("Lab settings")
    model_name = st.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = st.slider("Chunk size", 300, 1800, DEFAULT_CHUNK_SIZE, 50)
    overlap = st.slider("Chunk overlap", 0, 400, DEFAULT_CHUNK_OVERLAP, 25)
    mode = st.selectbox("Retrieval", ["Hybrid", "Dense", "BM25"])
    top_k = st.slider("Top-k passages", 1, 12, 5)
    rerank = st.checkbox("Cross-encoder reranking")
    selected = st.multiselect("Filter sources", sources, default=sources)
    groq_key = st.text_input("Groq API key (optional)", os.getenv("GROQ_API_KEY", ""), type="password")
    groq_model = st.text_input("Groq model", "llama-3.3-70b-versatile")
    if st.button("Clear my documents and chat"):
        st.session_state.index = {"chunks": [], "created_at": None}
        st.session_state.messages = []
        st.session_state.notice = ""
        st.rerun()
    st.caption("Documents stay in this temporary session only.")

active = [item for item in chunks if item.get("source") in selected]
chat_tab, docs_tab, eval_tab, about_tab = st.tabs(["Ask your PDFs", "Knowledge base", "Evaluation", "How it works"])

with chat_tab:
    st.subheader("Your document assistant")
    st.write("Upload one or more PDFs, then ask questions about their contents.")
    uploads = st.file_uploader("Drop PDF files here or browse", type=["pdf"], accept_multiple_files=True,
                               help=f"Up to {MAX_FILES} PDFs, {MAX_MB} MB each.", key="pdf_upload")
    if st.button("Process PDFs", type="primary", disabled=not uploads):
        try:
            if len(uploads) > MAX_FILES:
                raise ValueError(f"Maximum {MAX_FILES} PDFs per upload.")
            docs = []
            for uploaded in uploads:
                data = uploaded.getvalue()
                if len(data) > MAX_MB * 1024 * 1024:
                    raise ValueError(f"{uploaded.name} exceeds {MAX_MB} MB.")
                body, metadata = load_uploaded_file(uploaded.name, data)
                if not body.strip():
                    raise ValueError(f"No selectable text found in {uploaded.name}. Scanned PDFs need OCR.")
                if len(body) > 2_000_000:
                    raise ValueError(f"{uploaded.name} contains too much text for this demo.")
                docs.append((uploaded.name, body, metadata))
            with st.spinner("Extracting, chunking, and embedding PDFs..."):
                added = prepare_chunks(docs, get_embedder(model_name), chunk_size, overlap)
            if not added:
                raise ValueError("No text passages were created.")
            replace = {name for name, _, _ in docs}
            keep = [item for item in chunks if item.get("source") not in replace]
            st.session_state.index = {"chunks": keep + added, "created_at": time.time()}
            st.session_state.notice = f"Ready: {len(added)} passages from {len(docs)} PDF(s)."
            st.rerun()
        except Exception as exc:
            st.error(f"PDF processing failed: {exc}")
    if st.session_state.notice:
        st.success(st.session_state.notice)
    if not active:
        st.info("Upload a PDF above or load the free RAG learning pack in Knowledge base.")
    else:
        cols = st.columns(3)
        cols[0].metric("Passages", len(active))
        cols[1].metric("Documents", len({item.get("source") for item in active}))
        cols[2].metric("Retrieval", mode)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved evidence"):
                    show_evidence(message["sources"])
    question = st.chat_input("Ask a question about your PDFs...", disabled=not active)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                vector = get_embedder(model_name).encode([question])[0]
                if mode == "Dense":
                    results = dense_search(vector, active, top_k=top_k)
                elif mode == "BM25":
                    results = bm25_search(question, active, top_k=top_k)
                else:
                    results = hybrid_search(question, vector, active, top_k=top_k)
                if rerank and results:
                    results = rerank_results(question, results, top_k=top_k)
                answer = answer_question(question, results, api_key=groq_key, model=groq_model)
                st.markdown(answer["answer"])
                st.caption(f"{mode} retrieval · {len(results)} passages · {answer['mode']}")
                with st.expander("Inspect evidence", expanded=True):
                    show_evidence(results)
                st.session_state.messages.append({"role": "assistant", "content": answer["answer"], "sources": results})
            except Exception as exc:
                st.error(f"Could not answer: {exc}")

with docs_tab:
    st.subheader("Free RAG learning pack")
    st.write("Six curated notes explain RAG, chunking, retrieval, hybrid search, reranking, and evaluation.")
    if st.button("Load free learning pack", type="primary"):
        try:
            with st.spinner("Building your session knowledge base..."):
                docs = sample_documents()
                added = prepare_chunks(docs, get_embedder(model_name), chunk_size, overlap)
            replace = {name for name, _, _ in docs}
            keep = [item for item in chunks if item.get("source") not in replace]
            st.session_state.index = {"chunks": keep + added, "created_at": time.time()}
            st.session_state.notice = f"Loaded {len(docs)} notes into {len(added)} passages."
            st.success(st.session_state.notice)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load starter pack: {exc}")
    for source in sources:
        selected_chunks = [item for item in chunks if item.get("source") == source]
        with st.expander(f"{source} · {len(selected_chunks)} passages"):
            st.text(selected_chunks[0].get("text", "")[:1000])
            metadata = selected_chunks[0].get("metadata", {})
            if metadata.get("source_url"):
                st.markdown(f"[Original source]({metadata['source_url']})")

with eval_tab:
    st.subheader("Retrieval evaluation")
    eval_path = SAMPLE_DIR / "evaluation.json"
    if eval_path.exists():
        st.download_button("Download starter evaluation JSON", eval_path.read_text(encoding="utf-8"),
                           "rag_evaluation.json", "application/json")
    eval_file = st.file_uploader("Evaluation dataset", type=["json"], key="eval_upload")
    if eval_file and st.button("Run evaluation", disabled=not active):
        try:
            cases = json.loads(eval_file.getvalue().decode("utf-8"))
            rows = []
            embedder = get_embedder(model_name)
            for case in cases:
                vector = embedder.encode([case["question"]])[0]
                if mode == "Dense":
                    found = dense_search(vector, active, top_k=top_k)
                elif mode == "BM25":
                    found = bm25_search(case["question"], active, top_k=top_k)
                else:
                    found = hybrid_search(case["question"], vector, active, top_k=top_k)
                rows.append({"relevant_sources": case["relevant_sources"],
                             "retrieved_sources": [item.get("source") for item in found]})
            metrics = evaluate_retrieval(rows, k=top_k)
            cols = st.columns(5)
            for col, label, key in zip(cols, ["Hit Rate", "Recall", "Precision", "MRR", "nDCG"],
                                       ["hit_rate", "recall", "precision", "mrr", "ndcg"]):
                col.metric(f"{label}@{top_k}", f"{metrics[key]:.3f}")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

with about_tab:
    st.subheader("How the RAG pipeline works")
    st.markdown("""
    1. Upload a PDF into your private browser session.
    2. pypdf extracts readable text and page markers.
    3. LangChain splits the text into overlapping passages.
    4. Sentence Transformers creates embeddings.
    5. Dense, BM25, or hybrid retrieval finds relevant passages.
    6. LangChain connects a prompt to Groq for grounded answers.
    7. The app shows the evidence passages used for the answer.
    """)
    st.info("Each visitor has a separate in-memory index. Uploaded documents are not written to a shared file. Session data is temporary.")
    st.warning("This is an educational public demo, not a hardened production service. Add rate limits and abuse monitoring before large-scale use. Do not upload confidential files. Scanned PDFs need OCR.")
