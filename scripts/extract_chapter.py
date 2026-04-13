#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def _fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _sha256_png(image: Image.Image) -> str:
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return hashlib.sha256(bio.getvalue()).hexdigest()


def _clamp_rect_xyxy(rect, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(x) for x in rect]
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    # Deterministic integer rounding
    ix1 = int(round(x1))
    iy1 = int(round(y1))
    ix2 = int(round(x2))
    iy2 = int(round(y2))
    ix1 = max(0, min(w, ix1))
    ix2 = max(0, min(w, ix2))
    iy1 = max(0, min(h, iy1))
    iy2 = max(0, min(h, iy2))
    if ix2 <= ix1:
        ix2 = min(w, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(h, iy1 + 1)
    return ix1, iy1, ix2, iy2


def _center(bbox_xyxy) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(x) for x in bbox_xyxy]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _contains(panel_xyxy, text_xyxy) -> bool:
    cx, cy = _center(text_xyxy)
    x1, y1, x2, y2 = [float(x) for x in panel_xyxy]
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


def _speaker_for_text_idx(*, text_idx: int, text_to_char: dict[int, int], char_labels: list) -> str:
    char_idx = text_to_char.get(text_idx)
    if char_idx is None:
        return "unsure"
    if 0 <= char_idx < len(char_labels):
        return f"char_{char_labels[char_idx]}"
    return f"char_{char_idx}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Stage 1 (Magi-only): run Magi OCR + panel detection; crop each detected panel (yellow boxes) into numbered "
            "subfolders and write transcripts + a minimal final/storyboard.json (scene fields left empty)."
        )
    )
    ap.add_argument("--chapter-id", required=True)
    ap.add_argument("--images", nargs="+", required=True, help="Page image paths (or a folder containing images).")
    ap.add_argument("--out", required=True, help="Chapter output root folder (will contain final/...).")

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--model", default="ragavsachdeva/magiv3")
    ap.add_argument("--attn", default="eager", choices=["auto", "eager", "sdpa"])
    ap.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face downloads if files are missing locally (default: local-files-only).",
    )

    # Deprecated: Stage 2 moved to scripts/add_scenes.py (kept for backward compatibility).
    ap.add_argument(
        "--scene-provider",
        choices=["ollama", "gemini", "auto", "none"],
        default="none",
        help="DEPRECATED (ignored). Run `python scripts/add_scenes.py final/storyboard.json` instead.",
    )
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434", help=argparse.SUPPRESS)
    ap.add_argument("--ollama-model", default="llava-phi3:latest", help=argparse.SUPPRESS)
    ap.add_argument("--gemini-model", default="gemini-2.5-flash", help=argparse.SUPPRESS)
    ap.add_argument("--gemini-key-env", default="GEMINI_API_KEY", help=argparse.SUPPRESS)
    ap.add_argument("--gemini-timeout", type=int, default=120, help=argparse.SUPPRESS)
    ap.add_argument("--gemini-batch-size", type=int, default=6, help=argparse.SUPPRESS)
    ap.add_argument("--gemini-thinking-budget", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--gemini-log-key", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cache", action="store_true", help=argparse.SUPPRESS)

    ap.add_argument("--max-panels", type=int, default=0, help="Optional cap per page (0 = no cap).")
    ap.add_argument("--debug", action="store_true", help="Write extra debug outputs under debug/.")

    args = ap.parse_args()

    if getattr(args, "scene_provider", "none") != "none":
        print("Warning: --scene-provider is deprecated and ignored in Stage 1. Run scripts/add_scenes.py for scenes.")

    if not args.allow_downloads:
        # Avoid network calls in restricted environments and keep runs reproducible.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    out_root = Path(args.out).expanduser()
    final_root = out_root / "final"
    pages_dir = final_root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the demo implementation for model + providers (keeps behavior consistent).
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import examples.magiv3_demo as demo  # type: ignore

    image_paths = demo._collect_image_paths(args.images)
    device = demo._pick_device(args.device)
    dtype = demo._dtype_for_device(device)
    attn_implementation = None if args.attn == "auto" else args.attn

    print(f"Loading model={args.model!r} on device={device!r} dtype={str(dtype).replace('torch.', '')}...")
    model = demo.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        local_files_only=not bool(args.allow_downloads),
    ).to(device).eval()
    processor = demo.AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=not bool(args.allow_downloads),
    )

    # Process one page at a time to keep peak memory low (important on MPS).
    det_assoc = []
    ocr = []
    with demo.torch.no_grad():
        for i, p in enumerate(image_paths):
            print(f"[Magi] Processing page {i+1}/{len(image_paths)}: {p}")
            img_np = demo._read_image_rgb_np(p)
            det = model.predict_detections_and_associations([img_np], processor)
            oc = model.predict_ocr([img_np], processor)
            det_assoc.append(det[0] if isinstance(det, list) and det else det)
            ocr.append(oc[0] if isinstance(oc, list) and oc else oc)

    # Optional debug: annotated overlays + per-page raw JSON
    if args.debug:
        debug_root = out_root / "debug"
        debug_root.mkdir(parents=True, exist_ok=True)
        for page_idx, (img_path, page_result, page_ocr) in enumerate(zip(image_paths, det_assoc, ocr)):
            page_result = page_result if isinstance(page_result, dict) else {"result": page_result}
            page_out = {
                "image_path": img_path,
                "detections_and_associations": page_result,
                "ocr_raw": page_ocr,
            }
            (debug_root / f"page_{page_idx}.json").write_text(
                json.dumps(demo._to_jsonable(page_out), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            demo._draw_overlay(img_path, page_result, str(debug_root / f"page_{page_idx}_annotated.png"))

    panels_out: list[dict] = []
    chapter_slug = str(args.chapter_id).strip().replace("/", "__")
    if not chapter_slug:
        _fail("--chapter-id must not be empty.")

    for page_idx, (img_path, page_result, page_ocr) in enumerate(zip(image_paths, det_assoc, ocr)):
        page_result = page_result if isinstance(page_result, dict) else {"result": page_result}

        panels_raw = page_result.get("panels", []) or []
        if not isinstance(panels_raw, list):
            panels_raw = []

        panels_xyxy: list[list[float]] = []
        for p in panels_raw:
            try:
                _, rect = demo._maybe_extract_polygon(p)
            except Exception:
                rect = None
            if rect is None:
                continue
            x1, y1, x2, y2 = [float(x) for x in rect]
            panels_xyxy.append([x1, y1, x2, y2])

        panels_xyxy = sorted(panels_xyxy, key=lambda r: (float(r[1]), float(r[0])))
        if int(args.max_panels) > 0:
            panels_xyxy = panels_xyxy[: int(args.max_panels)]

        ocr_texts = demo._extract_ocr_texts(page_ocr)
        ocr_bboxes = []
        if isinstance(page_ocr, dict) and isinstance(page_ocr.get("bboxes"), list):
            ocr_bboxes = page_ocr.get("bboxes") or []
        ocr_bboxes = ocr_bboxes if isinstance(ocr_bboxes, list) else []

        texts = page_result.get("texts", []) or []
        essential = page_result.get("is_essential_text", []) or []
        assoc = page_result.get("text_character_associations", []) or []
        char_labels = page_result.get("character_cluster_labels", []) or []

        text_to_char: dict[int, int] = {}
        if isinstance(assoc, list):
            for pair in assoc:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    try:
                        text_to_char[int(pair[0])] = int(pair[1])
                    except Exception:
                        continue

        pil_page = Image.open(img_path).convert("RGB")
        w, h = pil_page.size

        per_panel_meta: list[dict] = []
        for local_idx, rect in enumerate(panels_xyxy):
            ix1, iy1, ix2, iy2 = _clamp_rect_xyxy(rect, w, h)
            crop = pil_page.crop((ix1, iy1, ix2, iy2))
            crop_hash = _sha256_png(crop)
            pid = f"{chapter_slug}_p{page_idx:03d}_n{local_idx:03d}_{crop_hash[:10]}"

            panel_dir_rel = f"final/pages/{page_idx:03d}/panels/{local_idx:03d}"
            panel_dir = out_root / panel_dir_rel
            panel_dir.mkdir(parents=True, exist_ok=True)

            crop_rel = f"{panel_dir_rel}/panel.png"
            crop_path = out_root / crop_rel
            crop.save(crop_path)

            transcript_items: list[dict] = []
            ocr_lines: list[dict] = []
            n = max(
                len(texts) if isinstance(texts, list) else 0,
                len(ocr_texts),
                len(essential) if isinstance(essential, list) else 0,
            )
            for text_idx in range(n):
                raw_text = ocr_texts[text_idx] if text_idx < len(ocr_texts) else ""
                if not isinstance(raw_text, str):
                    raw_text = str(raw_text)
                text = raw_text.strip()
                if not text:
                    continue

                bbox = None
                if isinstance(texts, list) and text_idx < len(texts):
                    try:
                        bbox = demo._safe_rect_xyxy(texts[text_idx])
                    except Exception:
                        bbox = None
                if bbox is None and text_idx < len(ocr_bboxes):
                    bb = ocr_bboxes[text_idx]
                    if isinstance(bb, (list, tuple)) and len(bb) == 4:
                        bbox = tuple(float(x) for x in bb)

                if bbox is None or not _contains(rect, bbox):
                    continue

                is_ess = True
                if isinstance(essential, list) and text_idx < len(essential):
                    try:
                        is_ess = bool(essential[text_idx])
                    except Exception:
                        is_ess = True

                speaker = _speaker_for_text_idx(
                    text_idx=text_idx,
                    text_to_char=text_to_char,
                    char_labels=char_labels if isinstance(char_labels, list) else [],
                )
                bbox_xyxy = [float(x) for x in bbox]
                transcript_items.append(
                    {
                        "text_idx": int(text_idx),
                        "speaker": speaker,
                        "text": text,
                        "bbox": bbox_xyxy,
                        "essential": bool(is_ess),
                    }
                )
                ocr_lines.append({"text": text, "bbox": bbox_xyxy, "speaker": speaker})

            transcript_lines = [f"<{it['speaker']}>: {it['text']}" for it in transcript_items]
            (panel_dir / "transcript.txt").write_text(
                "\n".join(transcript_lines) + ("\n" if transcript_lines else ""),
                encoding="utf-8",
            )
            (panel_dir / "transcript.json").write_text(
                json.dumps(transcript_items, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (panel_dir / "panel.json").write_text(
                json.dumps(
                    {
                        "panel_id": pid,
                        "page_idx": int(page_idx),
                        "panel_idx": int(local_idx),
                        "bbox": [float(x) for x in rect],
                        "crop_path": crop_rel,
                        "crop_sha256_png": crop_hash,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            meta = {
                "panel_id": pid,
                "page_idx": int(page_idx),
                "bbox": [float(x) for x in rect],
                "crop_path": crop_rel,
                "ocr_lines": [x for x in ocr_lines if x.get("text")],
                "scene_caption": "",
                "scene_tags": [],
            }
            per_panel_meta.append(meta)

        panels_out.extend(per_panel_meta)

    storyboard = {
        "version": "v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "chapter_id": str(args.chapter_id),
        "source_images": [str(p) for p in image_paths],
        "panels": panels_out,
        "beats": [],
        "script": [],
    }

    storyboard_path = final_root / "storyboard.json"
    storyboard_path.write_text(json.dumps(demo._to_jsonable(storyboard), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {storyboard_path}")
    print(f"- panels: {len(panels_out)}")
    print(f"- crops: {pages_dir}")


if __name__ == "__main__":
    main()
