from __future__ import annotations

from raglab.chunking import chunk_text


def prepare_chunks(documents: list[tuple[str, str, dict]], embedder, chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    """Turn (source name, text, metadata) documents into embedded chunks."""
    pending: list[dict] = []
    for source, text, metadata in documents:
        for chunk_number, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            pending.append({
                "text": piece,
                "source": source,
                "chunk_id": f"{source}:{chunk_number}",
                "metadata": {**metadata, "chunk_index": chunk_number},
            })
    if not pending:
        return []
    vectors = embedder.encode([item["text"] for item in pending])
    if len(vectors) != len(pending):
        raise RuntimeError("Embedding model returned a different number of vectors than input chunks.")
    for item, vector in zip(pending, vectors):
        item["embedding"] = vector
    return pending
