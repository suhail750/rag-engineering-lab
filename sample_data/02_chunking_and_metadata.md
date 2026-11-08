# Ingestion, chunking, metadata, and embeddings

## Ingestion
Ingestion brings data into a RAG system. Sources may include PDF, HTML, Word, Markdown, CSV, or databases. Extract useful text, remove noise, preserve structure, and record where each passage came from.

Useful metadata includes document title, URL, author, date, page, section, and access permissions. Metadata supports filtering, citations, freshness, and access control.

## Chunking
Chunking divides long documents into smaller passages called chunks. A whole book is too large and broad to retrieve as one unit; focused chunks are easier to match to a question.

- **Fixed-size:** split at a chosen token or character limit. Simple, but can cut ideas awkwardly.
- **Recursive:** split at natural boundaries such as headings, paragraphs, and sentences. A useful general default.
- **Semantic:** use meaning or topic changes to choose boundaries. More complex, but can preserve coherent ideas.
- **Structure-aware:** split by HTML headings, Markdown sections, PDF pages, or code functions.
- **Parent-child:** retrieve a small child chunk, then provide its larger parent section for context.

## Chunk size and overlap
Small chunks can be precise but lose context. Large chunks preserve context but may include unrelated material and consume more tokens. Overlap repeats some text between adjacent chunks to preserve ideas crossing boundaries. Too much overlap creates duplicates and noise. There is no universal best size: evaluate with real questions.

## Embeddings
An embedding model converts text into a vector (a list of numbers). Related meanings often have nearby vectors. The question is embedded using a compatible model and compared with document vectors.

Embeddings help semantic search: "find passages about understanding similar meanings" may retrieve a passage about semantic retrieval even if the exact words differ.

Limitations: similarity does not guarantee that a passage answers the question; domain terms and language affect quality; query and document vectors must use compatible models and dimensions; changing the model usually requires re-embedding the corpus.

## Metadata filtering
Filters restrict retrieval by date, category, language, document type, or permissions. Enforce access-control filters in application logic, not merely in a prompt.

## Practical checklist
1. Preserve original source URLs and stable document IDs.
2. Remove repeated headers, footers, and extraction noise.
3. Keep headings with the text they explain.
4. Store source metadata on every chunk.
5. Inspect real chunks before selecting chunk size.
6. Evaluate with representative user questions.

Source: https://www.mongodb.com/docs/vector-search/tutorials/rag/
