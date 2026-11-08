from __future__ import annotations

import csv
import io
import json
from pathlib import Path


def load_uploaded_file(filename: str, content: bytes) -> tuple[str, dict]:
    """Extract text from supported user-uploaded formats."""
    suffix = Path(filename).suffix.lower()
    metadata = {"filename": filename, "file_type": suffix.lstrip(".")}
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[Page {number}]\n{page_text}")
        return "\n\n".join(pages), {**metadata, "page_count": len(reader.pages)}
    if suffix == ".docx":
        from docx import Document
        document = Document(io.BytesIO(content))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                paragraphs.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(paragraphs), metadata
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8-sig", errors="replace"), metadata
    if suffix == ".csv":
        rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig", errors="replace"))))
        return "\n".join(" | ".join(row) for row in rows), metadata
    if suffix == ".json":
        parsed = json.loads(content.decode("utf-8-sig"))
        return json.dumps(parsed, ensure_ascii=False, indent=2), metadata
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")
