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
import requests


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: extract per-panel crops + metadata into a minimal storyboard.json.")
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

    ap.add_argument("--scene-provider", choices=["ollama", "gemini", "auto", "none"], default="auto")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", default="llava-phi3:latest")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    ap.add_argument("--gemini-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--gemini-timeout", type=int, default=120)
    ap.add_argument("--gemini-batch-size", type=int, default=6)
    ap.add_argument("--gemini-thinking-budget", type=int, default=0)
    ap.add_argument("--gemini-log-key", action="store_true")

    ap.add_argument("--max-panels", type=int, default=0, help="Optional cap per page (0 = no cap).")
    ap.add_argument("--cache", action="store_true", help="Enable on-disk Gemini scene cache under final/.cache/.")
    ap.add_argument("--debug", action="store_true", help="Write extra debug outputs under debug/.")

    args = ap.parse_args()

    if not args.allow_downloads:
        # Avoid network calls in restricted environments and keep runs reproducible.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    out_root = Path(args.out).expanduser()
    final_root = out_root / "final"
    panels_dir = final_root / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = final_root / ".cache"
    cache_path = cache_dir / "scene_cache_gemini.json"
    if args.cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

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

    images_np = [demo._read_image_rgb_np(p) for p in image_paths]
    with demo.torch.no_grad():
        det_assoc = model.predict_detections_and_associations(images_np, processor)
        ocr = model.predict_ocr(images_np, processor)

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

    # Scene caption provider setup (only if requested)
    scene_provider = args.scene_provider
    gemini_rotator = None
    gemini_cache: dict = {}
    if scene_provider in {"gemini", "auto"}:
        gemini_keys = demo._load_gemini_keys(args.gemini_key_env)
        if gemini_keys:
            if args.gemini_log_key:
                fps = [hashlib.sha256(k.encode("utf-8")).hexdigest()[:10] for k in gemini_keys]
                print(f"[Gemini] Key env base name: {args.gemini_key_env}")
                print(f"[Gemini] Key fingerprints (index → fingerprint): {list(enumerate(fps))}")
            gemini_rotator = demo.GeminiKeyRotator(gemini_keys, log_key_fingerprint=args.gemini_log_key)
            if args.cache and cache_path.exists():
                try:
                    gemini_cache = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    gemini_cache = {}
            gemini_cache = gemini_cache if isinstance(gemini_cache, dict) else {}
        elif scene_provider == "gemini":
            _fail(f"No Gemini keys found in env var {args.gemini_key_env!r} (+ _2.._5).")

    panels_out: list[dict] = []
    prompt_version = "v1"
    chapter_slug = str(args.chapter_id).strip().replace("/", "__")
    if not chapter_slug:
        _fail("--chapter-id must not be empty.")

    for page_idx, (img_path, page_result, page_ocr) in enumerate(zip(image_paths, det_assoc, ocr)):
        page_result = page_result if isinstance(page_result, dict) else {"result": page_result}

        panels_xyxy = page_result.get("panels", []) or []
        if not isinstance(panels_xyxy, list):
            panels_xyxy = []
        panels_xyxy = [p for p in panels_xyxy if isinstance(p, (list, tuple)) and len(p) == 4]
        panels_xyxy = sorted(panels_xyxy, key=lambda r: (float(r[1]), float(r[0])))
        if int(args.max_panels) > 0:
            panels_xyxy = panels_xyxy[: int(args.max_panels)]

        ocr_texts = []
        ocr_bboxes = []
        if isinstance(page_ocr, dict):
            ocr_texts = page_ocr.get("ocr_texts") or []
            ocr_bboxes = page_ocr.get("bboxes") or []
        if not (isinstance(ocr_texts, list) and isinstance(ocr_bboxes, list) and len(ocr_texts) == len(ocr_bboxes)):
            ocr_texts, ocr_bboxes = [], []

        pil_page = Image.open(img_path).convert("RGB")
        w, h = pil_page.size

        # Gemini batching needs images for just the missing cache keys
        gemini_batch: list[tuple[int, Image.Image]] = []
        gemini_batch_keys: dict[int, str] = {}

        per_panel_meta: list[dict] = []
        for local_idx, rect in enumerate(panels_xyxy):
            ix1, iy1, ix2, iy2 = _clamp_rect_xyxy(rect, w, h)
            crop = pil_page.crop((ix1, iy1, ix2, iy2))
            crop_hash = _sha256_png(crop)
            pid = f"{chapter_slug}_p{page_idx:03d}_n{local_idx:03d}_{crop_hash[:10]}"

            crop_rel = f"final/panels/{pid}.png"
            crop_path = out_root / crop_rel
            crop.save(crop_path)

            ocr_lines = []
            for t, bb in zip(ocr_texts, ocr_bboxes):
                if not isinstance(t, str):
                    continue
                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    continue
                if _contains(rect, bb):
                    ocr_lines.append({"text": t.strip(), "bbox": [float(x) for x in bb]})

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

            if scene_provider in {"gemini", "auto"} and gemini_rotator is not None:
                cache_key = f"{crop_hash}:{args.gemini_model}:{prompt_version}"
                cached = gemini_cache.get(cache_key) if isinstance(gemini_cache, dict) else None
                if isinstance(cached, dict) and isinstance(cached.get("caption"), str):
                    meta["scene_caption"] = (cached.get("caption") or "").strip()
                    tags = cached.get("tags") if isinstance(cached.get("tags"), list) else []
                    meta["scene_tags"] = [str(x).strip().lower() for x in tags if str(x).strip()]
                else:
                    gemini_batch.append((local_idx, crop))
                    gemini_batch_keys[local_idx] = cache_key

            elif scene_provider == "ollama":
                try:
                    caption = demo._ollama_generate_text(
                        host=args.ollama_host,
                        model=args.ollama_model,
                        prompt=demo._ollama_caption_prompt(),
                        image=crop,
                        max_tokens=128,
                        temperature=0.2,
                        timeout_s=120,
                    )
                except Exception as e:
                    caption = f"ERROR: {e}"
                meta["scene_caption"] = (caption or "").strip()
                try:
                    tags_text = demo._ollama_generate_text(
                        host=args.ollama_host,
                        model=args.ollama_model,
                        prompt=demo._ollama_tags_prompt(meta["scene_caption"]),
                        image=None,
                        max_tokens=64,
                        temperature=0.0,
                        timeout_s=120,
                    )
                    tags, _ = demo._parse_scene_labels(tags_text)
                    meta["scene_tags"] = tags
                except Exception:
                    pass

            per_panel_meta[-1] = meta

        # Run Gemini for uncached panels in this page
        if gemini_batch and gemini_rotator is not None:
            batch_size = max(1, int(args.gemini_batch_size))
            for i in range(0, len(gemini_batch), batch_size):
                batch = gemini_batch[i : i + batch_size]

                def _make_call():
                    def _call(api_key: str):
                        return demo._gemini_generate_scene_json_batch(
                            api_key=api_key,
                            model=args.gemini_model,
                            panel_images=batch,
                            max_tokens=max(256, 128 * len(batch)),
                            temperature=0.2,
                            thinking_budget=max(0, int(args.gemini_thinking_budget)),
                            timeout_s=max(5, int(args.gemini_timeout)),
                        )

                    return _call

                try:
                    items, raw = gemini_rotator.call(_make_call())
                except requests.exceptions.RequestException as e:
                    # Avoid leaking API keys in exception strings/URLs. Keep message minimal.
                    _fail(
                        "Gemini request failed (network/transport error). "
                        "Re-run with network access enabled. "
                        f"error_type={type(e).__name__}"
                    )

                # Apply results
                by_idx = {}
                for it in items or []:
                    try:
                        pidx = int(it.get("panel_idx"))
                    except Exception:
                        continue
                    by_idx[pidx] = it

                for local_idx, _ in batch:
                    meta = per_panel_meta[local_idx]
                    it = by_idx.get(local_idx)
                    if isinstance(it, dict):
                        meta["scene_caption"] = (it.get("caption") or "").strip()
                        tags = it.get("tags") if isinstance(it.get("tags"), list) else []
                        meta["scene_tags"] = [str(x).strip().lower() for x in tags if str(x).strip()]
                        ck = gemini_batch_keys.get(local_idx)
                        if args.cache and ck:
                            gemini_cache[ck] = {"caption": meta["scene_caption"], "tags": meta["scene_tags"]}
                    per_panel_meta[local_idx] = meta

        # Auto fallback for missing Gemini captions
        if scene_provider == "auto":
            for local_idx, meta in enumerate(per_panel_meta):
                if (meta.get("scene_caption") or "").strip() and not str(meta.get("scene_caption")).startswith("ERROR:"):
                    continue
                crop_rel = meta["crop_path"]
                crop_img = Image.open(out_root / crop_rel).convert("RGB")
                try:
                    caption = demo._ollama_generate_text(
                        host=args.ollama_host,
                        model=args.ollama_model,
                        prompt=demo._ollama_caption_prompt(),
                        image=crop_img,
                        max_tokens=128,
                        temperature=0.2,
                        timeout_s=120,
                    )
                except Exception as e:
                    caption = f"ERROR: {e}"
                meta["scene_caption"] = (caption or "").strip()
                try:
                    tags_text = demo._ollama_generate_text(
                        host=args.ollama_host,
                        model=args.ollama_model,
                        prompt=demo._ollama_tags_prompt(meta["scene_caption"]),
                        image=None,
                        max_tokens=64,
                        temperature=0.0,
                        timeout_s=120,
                    )
                    tags, _ = demo._parse_scene_labels(tags_text)
                    meta["scene_tags"] = tags
                except Exception:
                    pass
                per_panel_meta[local_idx] = meta

        panels_out.extend(per_panel_meta)

    if args.cache and gemini_rotator is not None:
        try:
            cache_path.write_text(json.dumps(demo._to_jsonable(gemini_cache), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

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
    print(f"- crops: {panels_dir}")


if __name__ == "__main__":
    main()
