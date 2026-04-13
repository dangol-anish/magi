from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    # Strip simple markdown artifacts that often appear in recaps.
    text = text.replace("`", "").strip()
    text = _WS_RE.sub(" ", text)
    return text.strip()

