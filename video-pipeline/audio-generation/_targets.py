from __future__ import annotations


def iter_targets(data: dict, *, include_pages: bool, include_panels: bool):
    pages = data.get("pages")
    if not isinstance(pages, list):
        return

    for page in pages:
        if not isinstance(page, dict):
            continue

        page_idx = page.get("page_idx")
        page_idx_i = page_idx if isinstance(page_idx, int) else None

        if include_pages:
            recap = page.get("recap")
            if isinstance(recap, str) and recap.strip():
                yield ("page_recap", page, page_idx_i, recap)

        if include_panels:
            panels = page.get("panels")
            if not isinstance(panels, list):
                continue
            for panel in panels:
                if not isinstance(panel, dict):
                    continue
                sentence = panel.get("sentence")
                if isinstance(sentence, str) and sentence.strip():
                    yield ("panel_sentence", panel, page_idx_i, sentence)


def count_targets(
    data: dict,
    *,
    include_pages: bool,
    include_panels: bool,
    only_page_idx: int | None,
) -> int:
    n = 0
    for _kind, _obj, page_idx, text in iter_targets(data, include_pages=include_pages, include_panels=include_panels):
        if only_page_idx is not None and page_idx is not None and page_idx != only_page_idx:
            continue
        if only_page_idx is not None and page_idx is None:
            continue
        if isinstance(text, str) and text.strip():
            n += 1
    return n

