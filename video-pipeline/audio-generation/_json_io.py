from __future__ import annotations

import json
from pathlib import Path

from _errors import fail


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"Failed to parse JSON: {path} ({e})")
    if not isinstance(data, dict):
        fail(f"Expected top-level object in {path}")
    return data


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

