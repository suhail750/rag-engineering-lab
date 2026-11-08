from __future__ import annotations

import json
from pathlib import Path

from raglab.config import INDEX_PATH


def load_index(path: Path = INDEX_PATH) -> dict:
    if not path.exists():
        return {"chunks": [], "created_at": None}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict) or not isinstance(value.get("chunks", []), list):
            raise ValueError("Index file has an unexpected structure.")
        return {"chunks": value.get("chunks", []), "created_at": value.get("created_at")}
    except (json.JSONDecodeError, OSError, ValueError):
        return {"chunks": [], "created_at": None}


def save_index(index: dict, path: Path = INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False)
    temporary.replace(path)
