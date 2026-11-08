import json
import os
from pathlib import Path

import streamlit as st

from raglab.config import APP_TITLE, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from raglab.embeddings import EmbeddingService
from raglab.evaluation import evaluate_retrieval, unique_sources_in_rank_order
from raglab.generation import answer_question
from raglab.ingestion import prepare_chunks
from raglab.retrieval import bm25_search, dense_search, hybrid_search, rerank_results

# Keep paths simple: all learning material lives in sample_data/.
ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT_DIR / "sample_data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title=APP_TITLE, page_icon="🏃", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1380px;}
.hero {padding: 2rem; border: 1px solid rgba(139,124,255,.35); border-radius: 22px;
       background: linear-gradient(120deg, rgba(139,124,255,.16), rgba(35,42,61,.35));}
.hero h1 {font-size: 2.6rem; margin-bottom: .5rem;}
.hero p {color: #C7CBD6; font-size: 1.05rem;}
</style>
<div class="hero">
  <h1>🏃 Sports Health & Athlete Safety RAG</h1>
  <p>Evidence-based sports injury, prevention, recovery, and athlete-safety information from public sports-medicine sources.</p>
</div>
""", unsafe_allow_html=True)


def get_streamlit_secret():
    try:
        return st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        return ""


def load_web_knowledge():
    """Read our curated notes and their original public web sources."""
    manifest = json.loads((SAMPLE_DIR / "sources.json").read_text(encoding="utf-8"))
    documents = []

    for item in manifest["documents"]:
        file_path = SAMPLE_DIR / item["file"]
        text = file_path.read_text(encoding="utf-8")

        metadata = {
            "title": item.get("title", item["file"]),
            "source_url": item.get("source_url", ""),
            "publisher": item.get("publisher", ""),
            "source_type": item.get("source_type", "web reference"),
        }
        documents.append((item["file"], text, metadata))

    return documents


@st.cache_resource(show_spinner="Loading the RAG knowledge base...")
def get_embedder():
    # Downloaded once and reused by the app.
    return EmbeddingService(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Preparing web-sourced knowledge for search...")
def get_knowledge_chunks():
    # LangChain splits each source note into searchable passages.
    documents = load_web_knowledge()
    return prepare_chunks(
        documents,
        get_embedder(),
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
    )


def show_evidence(results):
    """Show the exact passages that were retrieved for an answer."""
    for rank, item in enumerate(results, 1):
        st.markdown(
            f"**[{rank}] {item.get('source', 'Unknown')}** · "
            f"score: {item.get('score', 0):.4f}"
        )
        st.write(item.get("text", ""))

        metadata = item.get("metadata", {})
        source_url = metadata.get("source_url")
        if source_url:
            publisher = metadata.get("publisher") or "Original web source"
            st.markdown(f"[Read the original {publisher} material]({source_url})")


# The corpus is read-only and shared safely; each visitor has their own chat history.
chunks = get_knowledge_chunks()
sources = sorted({item.get("source", "unknown") for item in chunks})

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Sports RAG settings")
    retrieval_mode = st.selectbox("Retrieval method", ["Hybrid", "Dense", "BM25"])
    top_k = st.slider("Passages to retrieve", 1, 12, 5)
    use_reranker = st.checkbox("Use cross-encoder reranking")
    selected_sources = st.multiselect(
        "Limit search to sources",
        sources,
        default=sources,
    )
    gemini_key = st.text_input(
        "Gemini API key (optional)",
        os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "") or get_streamlit_secret(),
        type="password",
    )
    gemini_model = st.text_input("Gemini model", "gemini-3.5-flash-lite")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("This assistant answers from the shared sports-health knowledge base. Add GEMINI_API_KEY in Streamlit Secrets for generated explanations.")

active_chunks = [
    item for item in chunks
    if item.get("source", "unknown") in selected_sources
]

ask_tab, knowledge_tab, evaluation_tab, about_tab = st.tabs(
    ["Ask Sports RAG", "Sports Health Library", "Evaluation", "How it works"]
)

with ask_tab:
    st.subheader("Ask about sports health and athlete safety")
    st.write(
        "Ask about common sports injuries, ankle and knee injuries, muscle strains, concussion, heat illness, injury prevention, rehabilitation principles, and return to sport. "
        "Answers are grounded in retrieved passages from this knowledge base."
    )

    st.warning("Learning assistant: answers can be incomplete. Inspect the retrieved evidence and original source links.")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Knowledge passages", len(active_chunks))
    metric_cols[1].metric("Source notes", len(selected_sources))
    metric_cols[2].metric("Retrieval", retrieval_mode)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Inspect retrieved evidence"):
                    show_evidence(message["sources"])

    question = st.chat_input("For example: What should happen if a player is suspected of having a concussion?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                query_vector = get_embedder().encode([question])[0]
                # Retrieve a wider candidate pool before the slower cross-encoder stage.
                retrieval_k = max(top_k * 3, 12) if use_reranker else top_k

                if retrieval_mode == "Dense":
                    results = dense_search(query_vector, active_chunks, top_k=retrieval_k)
                elif retrieval_mode == "BM25":
                    results = bm25_search(question, active_chunks, top_k=retrieval_k)
                else:
                    results = hybrid_search(
                        question,
                        query_vector,
                        active_chunks,
                        top_k=retrieval_k,
                    )

                if use_reranker and results:
                    results = rerank_results(question, results, top_k=top_k)

                response = answer_question(
                    question,
                    results,
                    api_key=gemini_key,
                    model=gemini_model,
                )

                st.markdown(response["answer"])
                st.caption(
                    f"{retrieval_mode} retrieval · {len(results)} passages · "
                    f"{response['mode']}"
                )

                with st.expander("Inspect retrieved evidence", expanded=True):
                    show_evidence(results)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": results,
                })

            except Exception as error:
                st.error(f"Something went wrong while answering: {error}")


with knowledge_tab:
    st.subheader("Sports health knowledge library")
    st.write(
        "These are educational summaries based on public sports-medicine guidance, "
        "clinical education, and consensus statements. Open the original links to read the sources."
    )

    manifest = json.loads((SAMPLE_DIR / "sources.json").read_text(encoding="utf-8"))
    for source in manifest["documents"]:
        with st.expander(source.get("title", source["file"])):
            st.caption(f"File: {source['file']}")
            st.caption(f"Publisher: {source.get('publisher', 'Public source')}")
            st.markdown(f"[Open original web source]({source.get('source_url', '')})")

    st.divider()
    st.markdown("**Topics covered**")
    st.markdown(
        "- Common sports injuries, symptoms, and urgent warning signs\n"
        "- Ankle sprains and knee injuries, including ACL injuries\n"
        "- Muscle strains, tendon pain, and overuse injuries\n"
        "- Concussion recognition and return-to-sport progression\n"
        "- Heat illness, hydration, and athlete safety\n"
        "- Injury prevention, rehabilitation principles, and return to sport"
    )


with evaluation_tab:
    st.subheader("Test retrieval quality")
    st.write(
        "Run the evaluation questions included with this repository. "
        "This tests retrieval against labelled source notes; it does not grade the generated answer."
    )

    if st.button("Run the built-in evaluation", type="primary"):
        try:
            cases = json.loads(
                (SAMPLE_DIR / "evaluation.json").read_text(encoding="utf-8")
            )
            evaluation_rows = []

            for case in cases:
                question_text = case["question"]
                vector = get_embedder().encode([question_text])[0]
                retrieval_k = max(top_k * 3, 12) if use_reranker else top_k

                if retrieval_mode == "Dense":
                    found = dense_search(vector, active_chunks, top_k=retrieval_k)
                elif retrieval_mode == "BM25":
                    found = bm25_search(question_text, active_chunks, top_k=retrieval_k)
                else:
                    found = hybrid_search(
                        question_text,
                        vector,
                        active_chunks,
                        top_k=retrieval_k,
                    )

                # Evaluation labels are source-level, so count each source once.
                # Otherwise several chunks from one file can occupy the top-k slots.
                if use_reranker and found:
                    found = rerank_results(question_text, found, top_k=top_k)

                evaluation_rows.append({
                    "relevant_sources": case["relevant_sources"],
                    "retrieved_sources": unique_sources_in_rank_order(found, top_k),
                })

            metrics = evaluate_retrieval(evaluation_rows, k=top_k)

            columns = st.columns(5)
            metric_names = [
                ("Hit Rate", "hit_rate"),
                ("Recall", "recall"),
                ("Precision", "precision"),
                ("MRR", "mrr"),
                ("nDCG", "ndcg"),
            ]

            for column, (label, key) in zip(columns, metric_names):
                column.metric(f"{label}@{top_k}", f"{metrics[key]:.3f}")

        except Exception as error:
            st.error(f"Evaluation failed: {error}")


with about_tab:
    st.subheader("How this Sports Health RAG works")
    st.markdown("""
    1. **Web research:** we curate public sports-medicine, sports-health, and rehabilitation guidance.
    2. **Knowledge notes:** the sources are summarised into concise sports-health notes, with original URLs kept as metadata.
    3. **Chunking:** LangChain's text splitter breaks notes into searchable passages.
    4. **Embeddings:** Sentence Transformers converts passages and questions into vectors.
    5. **Retrieval:** Dense, BM25, or Hybrid search finds relevant passages.
    6. **Reranking:** an optional cross-encoder can reorder the retrieved passages.
    7. **Generation:** a LangChain prompt sends the question and evidence to Gemini.
    8. **Citations:** the app displays the retrieved passages and links to their original sources.
    """)

    st.info(
        "Without a Gemini API key, the app shows the most relevant retrieved passages. "
        "With a key, it generates an answer using only the retrieved context."
    )
    st.warning(
        "This is an educational information tool, not a medical device. It cannot diagnose injuries, prescribe treatment, or clear anyone to play. Seek professional care for injury-specific decisions and urgent care for serious warning signs."
    )
