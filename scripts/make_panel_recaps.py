#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def _fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_CROP_PATH_RE = re.compile(r"(?:^|/)final/pages/(\d+)/panels/(\d+)/")


def _panel_sort_key(panel: dict, fallback_idx: int) -> tuple[int, int, int]:
    crop_path = panel.get("crop_path")
    if isinstance(crop_path, str):
        m = _CROP_PATH_RE.search(crop_path.replace("\\", "/"))
        if m:
            try:
                return (int(m.group(1)), int(m.group(2)), fallback_idx)
            except Exception:
                pass
    page_idx = panel.get("page_idx")
    if isinstance(page_idx, int):
        try:
            n = int(page_idx)
        except Exception:
            n = 0
        return (n, fallback_idx, fallback_idx)
    return (fallback_idx, fallback_idx, fallback_idx)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_transcript(panel_dir: Path) -> str:
    txt = _read_text(panel_dir / "transcript.txt").strip()
    if txt:
        return txt

    # Fallback: transcript.json
    try:
        items = json.loads(_read_text(panel_dir / "transcript.json"))
    except Exception:
        items = None
    if isinstance(items, list):
        lines = []
        for it in items:
            if not isinstance(it, dict):
                continue
            speaker = it.get("speaker") if isinstance(it.get("speaker"), str) else "unsure"
            text = it.get("text") if isinstance(it.get("text"), str) else ""
            text = text.strip()
            if not text:
                continue
            lines.append(f"<{speaker}>: {text}")
        return "\n".join(lines).strip()

    return ""


def _prompt_for_panel(
    *,
    prev_context: str,
    panel_id: str,
    page_idx: int | None,
    scene_caption: str,
    scene_tags: list[str],
    transcript: str,
    sentences_min: int,
    sentences_max: int,
) -> str:
    tags_s = ", ".join([t.strip().lower() for t in scene_tags if isinstance(t, str) and t.strip()]) if scene_tags else ""
    transcript = transcript.strip()
    if transcript:
        # Prevent huge prompts from ultra-long OCR.
        lines = transcript.splitlines()
        transcript = "\n".join(lines[:40]).strip()

    return (
        "You are writing a coherent, sensible manga chapter recap as a narrator.\n"
        f"Write {sentences_min} to {sentences_max} sentences for THIS panel.\n\n"
        "Rules:\n"
        "- Maintain continuity with the previous recap context.\n"
        "- Use ONLY the provided evidence + what is clearly visible in the image.\n"
        "- Do not invent new character names or plot facts not supported by evidence.\n"
        "- Do not quote dialogue with quotation marks; paraphrase instead.\n"
        "- Write in simple, clear English.\n"
        "- Output ONLY the recap text.\n\n"
        f"Previous recap context:\n{prev_context.strip() or '(none)'}\n\n"
        f"Panel id: {panel_id}\n"
        + (f"Page index: {page_idx}\n" if page_idx is not None else "")
        + f"Scene caption: {scene_caption.strip() or '(none)'}\n"
        + f"Scene tags: {tags_s or '(none)'}\n"
        + f"Transcript (OCR):\n{transcript or '(none)'}\n"
    )


def _panel_block(*, panel_idx: int, panel_id: str, scene_caption: str, scene_tags: list[str], transcript: str) -> str:
    tags_s = ", ".join([t.strip().lower() for t in scene_tags if isinstance(t, str) and t.strip()]) if scene_tags else ""
    transcript = (transcript or "").strip()
    if transcript:
        lines = transcript.splitlines()
        transcript = "\n".join(lines[:25]).strip()
    return (
        f"[Sub-panel {panel_idx:03d}] panel_id={panel_id or '(unknown)'}\n"
        f"Scene caption: {scene_caption.strip() or '(none)'}\n"
        f"Scene tags: {tags_s or '(none)'}\n"
        f"Transcript (OCR):\n{transcript or '(none)'}\n"
    )


def _prompt_for_page(
    *,
    prev_context: str,
    page_idx: int,
    panel_blocks: list[str],
    sentences_min: int,
    sentences_max: int,
) -> str:
    blocks = "\n\n".join([b.strip() for b in panel_blocks if b.strip()]).strip()
    if blocks:
        # Prevent extremely large prompts.
        blocks = blocks[:12000]
    return (
        "You are writing a coherent, sensible manga chapter recap as a narrator.\n"
        f"Write {sentences_min} to {sentences_max} sentences for THIS page.\n\n"
        "Rules:\n"
        "- Maintain continuity with the previous recap context.\n"
        "- Use ONLY the provided evidence + what is clearly visible in the images.\n"
        "- Do not invent new character names or plot facts not supported by evidence.\n"
        "- Do not quote dialogue with quotation marks; paraphrase instead.\n"
        "- Write in simple, clear English.\n"
        "- Output ONLY the recap text.\n\n"
        f"Previous recap context:\n{prev_context.strip() or '(none)'}\n\n"
        f"Page index: {page_idx}\n\n"
        "Ordered sub-panels evidence:\n"
        + (blocks or "(none)")
        + "\n"
    )


def _parse_pages_spec(spec: str) -> set[int]:
    out: set[int] = set()
    for raw in (spec or "").split(","):
        s = raw.strip()
        if not s:
            continue
        if "-" in s:
            a, b = s.split("-", 1)
            a = a.strip()
            b = b.strip()
            if not a or not b:
                continue
            try:
                lo = int(a)
                hi = int(b)
            except Exception:
                continue
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(s))
            except Exception:
                continue
    return out


def _is_good_recap(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return not t.startswith("ERROR:")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Stage 3: generate recap text using Ollama.\n"
            "- mode=panel (default): recap per cropped sub-panel, written next to each crop\n"
            "- mode=page: one recap per page using all sub-panel crops + evidence, written to a chapter-level JSON"
        )
    )
    ap.add_argument("storyboard", help="Path to final/storyboard.json")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing recap outputs if present.")
    ap.add_argument(
        "--pages",
        default=None,
        help=(
            "Page indices to process (page mode only). Example: '0,1,2' or '0-5,7'. "
            "If provided, reruns only those pages and preserves others by merging with an existing recap_pages.json."
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Page mode only: if final/recap_pages.json exists, reuse existing pages and only regenerate pages "
            "that are missing/empty or have recap starting with 'ERROR:'."
        ),
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Page mode only: allow writing final/recap_pages.json containing only the selected --pages when no existing "
            "recap_pages.json is available to merge."
        ),
    )
    ap.add_argument("--context-panels", type=int, default=1, help="How many previous recaps to include as context.")
    ap.add_argument("--mode", choices=["panel", "page"], default="panel")

    ap.add_argument("--sentences-min", type=int, default=2)
    ap.add_argument("--sentences-max", type=int, default=4)

    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", default="qwen2.5vl:7b")
    ap.add_argument("--ollama-timeout", type=int, default=900, help="Ollama read timeout seconds.")
    ap.add_argument("--max-tokens", type=int, default=256, help="Max tokens for the generation call.")
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--progress", action="store_true", help="Print progress logs while processing.")
    ap.add_argument("--log-every", type=int, default=1, help="With --progress: print every N processed items.")

    # keep ollama up
    ap.add_argument("--keep-alive", default="60m", help="Ollama keep_alive duration (e.g. 60m, -1 for forever).")

    args = ap.parse_args()

    in_path = Path(args.storyboard)
    if not in_path.exists():
        _fail(f"File not found: {in_path}")

    final_root = in_path.parent
    out_root = final_root.parent

    doc = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        _fail("Top-level storyboard must be a JSON object.")
    panels = doc.get("panels")
    if not isinstance(panels, list) or not panels:
        _fail("storyboard.panels must be a non-empty array.")

    # Provider implementation via existing demo utilities.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import examples.magiv3_demo as demo  # type: ignore

    panels_sorted: list[dict] = []
    for idx, p in enumerate(panels):
        if isinstance(p, dict):
            pp = dict(p)
            pp["_order_idx"] = idx
            panels_sorted.append(pp)

    panels_sorted = sorted(panels_sorted, key=lambda p: _panel_sort_key(p, int(p.get("_order_idx") or 0)))

    mode = str(args.mode)
    # Use the user-provided --ollama-model as-is (no page-mode override).

    generated: list[dict] = []

    def _prev_context_for(out_dir: Path) -> str:
        k = max(0, int(args.context_panels))
        if k <= 0:
            return ""
        prev = [x.get("recap", "") for x in generated[-k:] if isinstance(x, dict)]
        prev = [str(x).strip() for x in prev if str(x).strip()]
        return "\n\n".join(prev).strip()

    # Count how many items will actually be processed (for progress).
    to_process = 0
    if mode == "panel":
        for p in panels_sorted:
            crop_rel = p.get("crop_path")
            if not isinstance(crop_rel, str) or not crop_rel.strip():
                continue
            crop_path = out_root / crop_rel
            if not crop_path.exists():
                continue
            panel_dir = crop_path.parent
            recap_txt = panel_dir / "recap.txt"
            if recap_txt.exists() and not bool(args.overwrite):
                continue
            to_process += 1
    else:
        # In page mode, we write a single chapter-level JSON output.
        chapter_json = final_root / "recap_pages.json"
        available_pages = sorted({int(p.get("page_idx")) for p in panels_sorted if isinstance(p.get("page_idx"), int)})
        pages_total = len(available_pages)
        selected_pages = available_pages
        if args.pages:
            spec = _parse_pages_spec(str(args.pages))
            selected_pages = [p for p in available_pages if p in spec]
        if chapter_json.exists() and not bool(args.overwrite) and not bool(args.resume) and not bool(args.pages):
            to_process = 0
        else:
            if bool(args.resume) and chapter_json.exists() and not bool(args.overwrite):
                # Best-effort estimate: count pages that are missing/errored in existing recap_pages.json.
                try:
                    existing = json.loads(chapter_json.read_text(encoding="utf-8"))
                except Exception:
                    existing = None
                existing_pages_by_idx: dict[int, dict] = {}
                if isinstance(existing, dict):
                    pages_in = existing.get("pages")
                    if isinstance(pages_in, list):
                        for it in pages_in:
                            if isinstance(it, dict) and isinstance(it.get("page_idx"), int):
                                existing_pages_by_idx[int(it["page_idx"])] = it
                bad = 0
                for page_idx in selected_pages:
                    prev = existing_pages_by_idx.get(int(page_idx))
                    prev_recap = prev.get("recap") if isinstance(prev, dict) else ""
                    if not _is_good_recap(str(prev_recap or "")):
                        bad += 1
                to_process = bad
            else:
                to_process = len(selected_pages)

    if args.progress:
        print(f"[Recap] host={args.ollama_host} model={args.ollama_model} mode={mode}")
        if mode == "panel":
            print(f"[Recap] panels to process: {to_process} / {len(panels_sorted)}")
        else:
            print(f"[Recap] pages to process: {to_process} / {pages_total}")
        if to_process:
            print("[Recap] Tip: first item may take a while while the model loads in Ollama.")
        sys.stdout.flush()

    # Warm-up: load model into memory and set keep_alive before processing starts.
    if args.progress:
        print(f"[Recap] warming up model (keep_alive={args.keep_alive})...")
        sys.stdout.flush()
    try:
        import requests
        requests.post(
            f"{args.ollama_host}/api/generate",
            json={
                "model": args.ollama_model,
                "prompt": "",
                "keep_alive": args.keep_alive,
            },
            timeout=120,
        )
    except Exception as e:
        if args.progress:
            print(f"[Recap] warm-up warning: {e}")
            sys.stdout.flush()

    start_all = time.time()
    processed = 0
    total = to_process

    if mode == "panel":
        for p in panels_sorted:
            crop_rel = p.get("crop_path")
            if not isinstance(crop_rel, str) or not crop_rel.strip():
                continue
            crop_path = out_root / crop_rel
            panel_dir = crop_path.parent
            if not crop_path.exists():
                continue

            recap_txt_path = panel_dir / "recap.txt"
            if recap_txt_path.exists() and not bool(args.overwrite):
                existing = _read_text(recap_txt_path).strip()
                if existing:
                    generated.append({"panel_id": p.get("panel_id", ""), "crop_path": crop_rel, "recap": existing})
                continue

            panel_id = p.get("panel_id") if isinstance(p.get("panel_id"), str) else ""
            page_idx = p.get("page_idx") if isinstance(p.get("page_idx"), int) else None
            scene_caption = p.get("scene_caption") if isinstance(p.get("scene_caption"), str) else ""
            scene_tags = p.get("scene_tags") if isinstance(p.get("scene_tags"), list) else []
            transcript = _read_transcript(panel_dir)

            prev_context = _prev_context_for(panel_dir)
            prompt = _prompt_for_panel(
                prev_context=prev_context,
                panel_id=panel_id or "(unknown)",
                page_idx=page_idx,
                scene_caption=scene_caption,
                scene_tags=scene_tags,
                transcript=transcript,
                sentences_min=max(1, int(args.sentences_min)),
                sentences_max=max(max(1, int(args.sentences_min)), int(args.sentences_max)),
            )

            processed += 1
            if args.progress and (processed == 1 or processed % max(1, int(args.log_every)) == 0 or processed == total):
                print(f"[Recap] ({processed}/{total}) generating for crop={crop_rel}")
                sys.stdout.flush()

            t0 = time.time()
            try:
                img = Image.open(crop_path).convert("RGB")
            except Exception as e:
                recap = f"ERROR: failed to open image: {e}"
                dt = time.time() - t0
                if args.progress:
                    print(f"[Recap] ({processed}/{total}) skipped (image error) in {dt:.1f}s")
                    sys.stdout.flush()
                continue

            try:
                recap = demo._ollama_generate_text(
                    host=args.ollama_host,
                    model=args.ollama_model,
                    prompt=prompt,
                    image=img,
                    max_tokens=max(32, int(args.max_tokens)),
                    temperature=float(args.temperature),
                    timeout_s=max(5, int(args.ollama_timeout)),
                )
            except Exception as e:
                recap = f"ERROR: {e}"

            recap = (recap or "").strip()
            dt = time.time() - t0
            if args.progress and (processed == 1 or processed % max(1, int(args.log_every)) == 0 or processed == total):
                print(f"[Recap] ({processed}/{total}) done in {dt:.1f}s")
                sys.stdout.flush()

            try:
                (panel_dir / "recap.txt").write_text(recap + ("\n" if recap else ""), encoding="utf-8")
                (panel_dir / "recap.json").write_text(
                    json.dumps(
                        {
                            "panel_id": panel_id,
                            "crop_path": crop_rel,
                            "recap": recap,
                            "model": args.ollama_model,
                            "provider": "ollama",
                            "generated_at": _now_utc_iso(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

            generated.append({"panel_id": panel_id, "crop_path": crop_rel, "recap": recap})

        # Chapter-level outputs (panel mode)
        script_lines: list[str] = []
        jsonl_lines: list[str] = []
        for p in panels_sorted:
            crop_rel = p.get("crop_path")
            if not isinstance(crop_rel, str) or not crop_rel.strip():
                continue
            crop_path = out_root / crop_rel
            panel_dir = crop_path.parent
            recap = _read_text(panel_dir / "recap.txt").strip()
            if not recap:
                continue
            script_lines.append(recap)
            panel_id = p.get("panel_id") if isinstance(p.get("panel_id"), str) else ""
            jsonl_lines.append(json.dumps({"panel_id": panel_id, "crop_path": crop_rel, "recap": recap}, ensure_ascii=False))

        (final_root / "recap_script.txt").write_text(
            "\n\n".join(script_lines).strip() + ("\n" if script_lines else ""),
            encoding="utf-8",
        )
        (final_root / "panel_recaps.jsonl").write_text(
            "\n".join(jsonl_lines).strip() + ("\n" if jsonl_lines else ""),
            encoding="utf-8",
        )
    else:
        chapter_json = final_root / "recap_pages.json"
        if chapter_json.exists() and not bool(args.overwrite) and not bool(args.resume) and not bool(args.pages):
            if args.progress:
                print(f"[Recap] exists (skip): {chapter_json}")
                print("[Recap] Tip: pass --overwrite to regenerate.")
                sys.stdout.flush()
            return

        existing_doc: dict | None = None
        existing_pages_by_idx: dict[int, dict] = {}
        if chapter_json.exists():
            try:
                existing_doc = json.loads(chapter_json.read_text(encoding="utf-8"))
            except Exception:
                existing_doc = None
            if isinstance(existing_doc, dict):
                pages_in = existing_doc.get("pages")
                if isinstance(pages_in, list):
                    for it in pages_in:
                        if not isinstance(it, dict):
                            continue
                        page_idx = it.get("page_idx")
                        if isinstance(page_idx, int):
                            existing_pages_by_idx[int(page_idx)] = dict(it)

        # Group by page and generate one recap per page using all sub-panel crops + evidence.
        by_page: dict[int, list[dict]] = {}
        for p in panels_sorted:
            page_idx = p.get("page_idx")
            if isinstance(page_idx, int):
                by_page.setdefault(int(page_idx), []).append(p)

        all_page_idxs = sorted(by_page.keys())
        selected_page_idxs = all_page_idxs
        if args.pages:
            spec = _parse_pages_spec(str(args.pages))
            selected_page_idxs = [p for p in all_page_idxs if p in spec]
            missing = sorted([p for p in spec if p not in set(all_page_idxs)])
            if args.progress and missing:
                print(f"[Recap] warning: requested pages not in storyboard: {missing}")
                sys.stdout.flush()
            if not chapter_json.exists() and not bool(args.allow_partial):
                _fail(
                    "final/recap_pages.json not found; cannot merge non-selected pages. "
                    "Run once without --pages (or pass --allow-partial to write a partial recap_pages.json)."
                )

        pages_out_by_idx: dict[int, dict] = {}
        # Seed with existing content if we are resuming or rerunning a subset and an output already exists.
        if existing_pages_by_idx and (bool(args.resume) or bool(args.pages)):
            for k, v in existing_pages_by_idx.items():
                if isinstance(k, int) and isinstance(v, dict):
                    pages_out_by_idx[int(k)] = dict(v)

        # Determine which pages actually need generation.
        pages_to_generate: list[int] = []
        if bool(args.overwrite):
            pages_to_generate = list(selected_page_idxs)
        elif bool(args.resume):
            for page_idx in selected_page_idxs:
                prev = pages_out_by_idx.get(int(page_idx))
                prev_recap = prev.get("recap") if isinstance(prev, dict) else ""
                if not _is_good_recap(str(prev_recap or "")):
                    pages_to_generate.append(int(page_idx))
        else:
            # If the chapter_json didn't exist, we need to generate the selected pages.
            if not chapter_json.exists():
                pages_to_generate = list(selected_page_idxs)
            else:
                # chapter_json exists and user specified --pages (handled by seeding + generating selected).
                pages_to_generate = list(selected_page_idxs)

        # If we have a pre-existing output and user requested --pages without overwrite,
        # treat it as "rerun these pages" (even if they were good).
        if bool(args.pages) and chapter_json.exists() and not bool(args.overwrite) and not bool(args.resume):
            pages_to_generate = list(selected_page_idxs)

        # Recompute total for progress display in page mode.
        total = len(pages_to_generate)
        processed = 0

        for page_idx in pages_to_generate:
            page_panels = by_page.get(page_idx) or []

            panel_blocks: list[str] = []
            images: list[Image.Image] = []
            used_panels: list[dict] = []

            for p in page_panels:
                crop_rel = p.get("crop_path")
                if not isinstance(crop_rel, str) or not crop_rel.strip():
                    continue
                crop_rel_norm = crop_rel.replace("\\", "/")
                m = _CROP_PATH_RE.search(crop_rel_norm)
                sub_idx = 0
                if m:
                    try:
                        sub_idx = int(m.group(2))
                    except Exception:
                        sub_idx = 0

                crop_path = out_root / crop_rel
                panel_dir = crop_path.parent
                panel_id = p.get("panel_id") if isinstance(p.get("panel_id"), str) else ""
                scene_caption = p.get("scene_caption") if isinstance(p.get("scene_caption"), str) else ""
                scene_tags = p.get("scene_tags") if isinstance(p.get("scene_tags"), list) else []
                transcript = _read_transcript(panel_dir)

                panel_blocks.append(
                    _panel_block(
                        panel_idx=sub_idx,
                        panel_id=panel_id,
                        scene_caption=scene_caption,
                        scene_tags=scene_tags,
                        transcript=transcript,
                    )
                )

                used_panels.append(
                    {
                        "sub_panel_idx": int(sub_idx),
                        "panel_id": panel_id,
                        "crop_path": crop_rel,
                    }
                )

                if crop_path.exists():
                    try:
                        images.append(Image.open(crop_path).convert("RGB"))
                    except Exception:
                        pass

            panel_blocks = [b for b in panel_blocks if b.strip()]
            panel_blocks = sorted(panel_blocks, key=lambda s: int(re.search(r"\[Sub-panel (\d+)\]", s).group(1)) if re.search(r"\[Sub-panel (\d+)\]", s) else 0)

            prev_context = _prev_context_for(final_root)
            prompt = _prompt_for_page(
                prev_context=prev_context,
                page_idx=page_idx,
                panel_blocks=panel_blocks,
                sentences_min=max(1, int(args.sentences_min)),
                sentences_max=max(max(1, int(args.sentences_min)), int(args.sentences_max)),
            )

            processed += 1
            if args.progress and (processed == 1 or processed % max(1, int(args.log_every)) == 0 or processed == total):
                print(f"[Recap] ({processed}/{total}) generating for page={page_idx:03d} sub_panels={len(page_panels)}")
                sys.stdout.flush()

            t0 = time.time()
            try:
                recap = demo._ollama_generate_text(
                    host=args.ollama_host,
                    model=args.ollama_model,
                    prompt=prompt,
                    image=images if images else None,
                    max_tokens=max(32, int(args.max_tokens)),
                    temperature=float(args.temperature),
                    timeout_s=max(5, int(args.ollama_timeout)),
                )
            except Exception as e:
                recap = f"ERROR: {e}"

            recap = (recap or "").strip()
            dt = time.time() - t0
            if args.progress and (processed == 1 or processed % max(1, int(args.log_every)) == 0 or processed == total):
                print(f"[Recap] ({processed}/{total}) done in {dt:.1f}s")
                sys.stdout.flush()

            used_panels = sorted([u for u in used_panels if isinstance(u, dict)], key=lambda u: int(u.get("sub_panel_idx") or 0))
            pages_out_by_idx[int(page_idx)] = {
                "page_idx": int(page_idx),
                "recap": recap,
                "panels": used_panels,
            }

        # Chapter-level output (page mode): JSON with linkage of recap -> panels used.
        if chapter_json.exists() and not bool(args.allow_partial):
            # Preserve all pages that exist in the storyboard, even if we only regenerated a subset.
            pages_out = [pages_out_by_idx.get(i) for i in all_page_idxs]
            pages_out = [x for x in pages_out if isinstance(x, dict)]
        else:
            # Partial mode: only include pages we have on hand (selected + any seeded pages).
            pages_out = [x for x in pages_out_by_idx.values() if isinstance(x, dict) and isinstance(x.get("page_idx"), int)]
            pages_out = sorted(pages_out, key=lambda x: int(x.get("page_idx") or 0))
        raw_script = "\n\n".join([str(x.get("recap") or "").strip() for x in pages_out if str(x.get("recap") or "").strip()]).strip()
        chapter_json.write_text(
            json.dumps(
                {
                    "mode": "page",
                    "storyboard": str(in_path),
                    "provider": "ollama",
                    "model": str(args.ollama_model),
                    "generated_at": _now_utc_iso(),
                    "raw_script": raw_script,
                    "pages": pages_out,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if args.progress:
        dt_all = time.time() - start_all
        if mode == "panel":
            print(f"[Recap] wrote: {final_root / 'recap_script.txt'}")
            print(f"[Recap] wrote: {final_root / 'panel_recaps.jsonl'}")
        else:
            print(f"[Recap] wrote: {final_root / 'recap_pages.json'}")
        print(f"[Recap] done in {dt_all/60.0:.1f} min")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
