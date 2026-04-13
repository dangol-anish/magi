from __future__ import annotations


def audio_path_for(*, kind: str, page_idx: int | None, panel_id: str | None) -> str:
    if kind == "page_recap":
        n = page_idx if isinstance(page_idx, int) else 0
        return f"audio/pages/page_{n:03d}_recap.wav"
    if kind == "panel_sentence":
        pid = (panel_id or "").strip() or "unknown_panel"
        return f"audio/panels/{pid}.wav"
    return "audio/unknown.wav"

