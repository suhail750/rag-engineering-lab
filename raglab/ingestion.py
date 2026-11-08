from __future__ import annotations
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def prepare_chunks(documents: list[tuple[str, str, dict]], embedder,
                   chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    """Split documents with LangChain, then add vectors for the simple retriever."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    langchain_docs = []
    for source, text, metadata in documents:
        if text.strip():
            langchain_docs.append(Document(
                page_content=text,
                metadata={"source": source, **metadata},
            ))
    split_docs = splitter.split_documents(langchain_docs)
    chunks = []
    for number, doc in enumerate(split_docs):
        chunks.append({
            "text": doc.page_content,
            "source": doc.metadata["source"],
            "chunk_id": f"{doc.metadata['source']}:{number}",
            "metadata": {**doc.metadata, "chunk_index": number,
                         "start_index": doc.metadata.get("start_index", 0)},
        })
    if not chunks:
        return []
    vectors = embedder.encode([item["text"] for item in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding model returned a different number of vectors than input chunks.")
    for item, vector in zip(chunks, vectors):
        item["embedding"] = vector
    return chunks
