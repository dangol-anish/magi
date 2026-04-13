#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import sys
import time
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


def _safe_gemini_error(e: Exception) -> str:
    if isinstance(e, requests.HTTPError):
        status = getattr(getattr(e, "response", None), "status_code", None)
        msg = ""
        try:
            body = (e.response.text or "").strip() if getattr(e, "response", None) is not None else ""
            data = json.loads(body) if body else {}
            err = data.get("error") if isinstance(data, dict) else None
            if isinstance(err, dict):
                m = err.get("message")
                s = err.get("status")
                msg = f"{s}: {m}" if s and m else (m or s or "")
        except Exception:
            msg = ""
        if status is not None and msg:
            return f"HTTP {status} {msg}"
        if status is not None:
            return f"HTTP {status}"
        return "HTTP error"
    if isinstance(e, requests.exceptions.RequestException):
        return f"{type(e).__name__}"
    return f"{type(e).__name__}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 2: add scene_caption + scene_tags to an existing final/storyboard.json (from Stage 1)."
    )
    ap.add_argument("storyboard", help="Path to final/storyboard.json")
    ap.add_argument("--out", default=None, help="Write to a different path instead of overwriting.")
    ap.add_argument("--overwrite", action="store_true", help="Replace existing scene_caption/scene_tags values.")

    ap.add_argument("--scene-provider", choices=["ollama", "gemini", "auto", "none"], default="auto")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", default="llava-phi3:latest")
    ap.add_argument(
        "--ollama-timeout", type=int, default=600,
        help="Ollama read timeout seconds (first run can be slow while the model loads).",
    )
    ap.add_argument("--ollama-caption-tokens", type=int, default=256,   # raised from 128
                    help="Max tokens for Ollama caption generation (default: 256).")
    ap.add_argument("--ollama-tags-tokens", type=int, default=64)
    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    ap.add_argument("--gemini-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--gemini-timeout", type=int, default=120)
    ap.add_argument("--gemini-batch-size", type=int, default=6)
    ap.add_argument("--gemini-thinking-budget", type=int, default=0)
    ap.add_argument("--gemini-log-key", action="store_true")
    ap.add_argument("--cache", action="store_true",
                    help="Enable on-disk Gemini scene cache under final/.cache/.")
    ap.add_argument("--progress", action="store_true",
                    help="Print progress logs while processing panels.")
    ap.add_argument("--log-every", type=int, default=1,
                    help="With --progress: print every N processed panels.")
    ap.add_argument(
        "--chapter-context",
        default="",
        help=(
            "Optional free-text context about this manga/chapter passed to the vision model. "
            "Example: 'Shounen action manga. Protagonist has spiky black hair.' "
            "Leave empty for fully automatic mode. Works for any manga series."
        ),
    )

    args = ap.parse_args()

    storyboard_path = Path(args.storyboard)
    if not storyboard_path.exists():
        _fail(f"File not found: {storyboard_path}")

    final_root = storyboard_path.parent
    out_root = final_root.parent
    cache_dir = final_root / ".cache"
    cache_path = cache_dir / "scene_cache_gemini.json"
    if args.cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import examples.magiv3_demo as demo  # type: ignore

    doc = json.loads(storyboard_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        _fail("Top-level storyboard must be a JSON object.")
    panels = doc.get("panels")
    if not isinstance(panels, list) or not panels:
        _fail("storyboard.panels must be a non-empty array.")

    scene_provider = args.scene_provider
    if scene_provider == "none":
        print("Scene provider set to 'none'; nothing to do.")
        return

    # ── Build character hints once from all panels ────────────────────────────
    # Uses cluster data already in the storyboard — no hardcoded names.
    # Works for any manga series.
    character_hints = demo._build_character_hints(
        panels,
        chapter_context=(args.chapter_context or "").strip(),
    )
    if args.progress and character_hints:
        print(f"[Stage2] Character hints built from storyboard:\n{character_hints}\n")
        sys.stdout.flush()

    # Build the caption prompt once (injecting hints)
    caption_prompt = demo._ollama_caption_prompt(character_hints=character_hints)

    prompt_version = "v2"   # bump when prompt changes to invalidate old Gemini cache

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

    def _needs_scene(panel: dict) -> bool:
        if args.overwrite:
            return True
        cap = panel.get("scene_caption")
        if isinstance(cap, str) and cap.strip():
            return False
        return True

    # ── Gemini pass ───────────────────────────────────────────────────────────
    if gemini_rotator is not None and scene_provider in {"gemini", "auto"}:
        gemini_batch: list[tuple[int, Image.Image]] = []
        gemini_batch_keys: dict[int, str] = {}

        for i, p in enumerate(panels):
            if not isinstance(p, dict):
                continue
            if not _needs_scene(p):
                continue
            crop_rel = p.get("crop_path")
            if not isinstance(crop_rel, str) or not crop_rel.strip():
                continue
            crop_path_full = out_root / crop_rel
            if not crop_path_full.exists():
                continue
            crop = Image.open(crop_path_full).convert("RGB")
            crop_hash = _sha256_png(crop)
            cache_key = f"{crop_hash}:{args.gemini_model}:{prompt_version}"

            cached = gemini_cache.get(cache_key) if isinstance(gemini_cache, dict) else None
            if isinstance(cached, dict) and isinstance(cached.get("caption"), str):
                p["scene_caption"] = (cached.get("caption") or "").strip()
                tags = cached.get("tags") if isinstance(cached.get("tags"), list) else []
                p["scene_tags"] = [str(x).strip().lower() for x in tags if str(x).strip()]
            else:
                gemini_batch.append((i, crop))
                gemini_batch_keys[i] = cache_key

        if gemini_batch:
            batch_size = max(1, int(args.gemini_batch_size))
            for start in range(0, len(gemini_batch), batch_size):
                batch = gemini_batch[start: start + batch_size]

                def _make_call(_hints: str = character_hints):
                    def _call(api_key: str):
                        return demo._gemini_generate_scene_json_batch(
                            api_key=api_key,
                            model=args.gemini_model,
                            panel_images=batch,
                            max_tokens=max(256, 128 * len(batch)),
                            temperature=0.2,
                            thinking_budget=max(0, int(args.gemini_thinking_budget)),
                            timeout_s=max(5, int(args.gemini_timeout)),
                            character_hints=_hints,   # ← injected here
                        )
                    return _call

                try:
                    items, _raw = gemini_rotator.call(_make_call())
                except Exception as e:
                    if scene_provider == "gemini":
                        _fail(f"Gemini request failed: {_safe_gemini_error(e)}")
                    err_s = _safe_gemini_error(e)
                    print(f"[Gemini] Warning: failed ({err_s}); will fall back to Ollama.")
                    for panel_idx, _ in batch:
                        p = panels[panel_idx]
                        if isinstance(p, dict):
                            p["scene_caption"] = f"ERROR: Gemini failed ({err_s})"
                            p["scene_tags"] = []
                    continue

                by_idx: dict[int, dict] = {}
                for it in items or []:
                    try:
                        pidx = int(it.get("panel_idx"))
                    except Exception:
                        continue
                    if isinstance(it, dict):
                        by_idx[pidx] = it

                for panel_idx, _ in batch:
                    p = panels[panel_idx]
                    if not isinstance(p, dict):
                        continue
                    it = by_idx.get(panel_idx)
                    if isinstance(it, dict):
                        p["scene_caption"] = (it.get("caption") or "").strip()
                        tags = it.get("tags") if isinstance(it.get("tags"), list) else []
                        p["scene_tags"] = [str(x).strip().lower() for x in tags if str(x).strip()]
                        ck = gemini_batch_keys.get(panel_idx)
                        if args.cache and ck:
                            gemini_cache[ck] = {
                                "caption": p["scene_caption"],
                                "tags": p.get("scene_tags") or [],
                            }
                    else:
                        if scene_provider == "auto":
                            p["scene_caption"] = "ERROR: Gemini returned no item for this panel"
                            p["scene_tags"] = []

    # ── Ollama pass ───────────────────────────────────────────────────────────
    if scene_provider in {"ollama", "auto"}:
        indices: list[int] = []
        for i, p in enumerate(panels):
            if not isinstance(p, dict):
                continue
            if not _needs_scene(p):
                continue
            cap = (p.get("scene_caption") or "").strip() if isinstance(p.get("scene_caption"), str) else ""
            if scene_provider == "auto" and cap and not cap.startswith("ERROR:"):
                continue
            crop_rel = p.get("crop_path")
            if not isinstance(crop_rel, str) or not crop_rel.strip():
                continue
            crop_path_full = out_root / crop_rel
            if not crop_path_full.exists():
                continue
            indices.append(i)

        if args.progress:
            print(f"[Ollama] host={args.ollama_host} model={args.ollama_model}")
            print(f"[Ollama] panels to process: {len(indices)} / {len(panels)}")
            if indices:
                print("[Ollama] Tip: first panel may take a while while the model loads.")
            sys.stdout.flush()

        processed = 0
        start_all = time.time()
        total = len(indices)

        for i in indices:
            p = panels[i]
            if not isinstance(p, dict):
                continue
            crop_rel = p.get("crop_path")
            if not isinstance(crop_rel, str) or not crop_rel.strip():
                continue
            crop_path_full = out_root / crop_rel
            crop = Image.open(crop_path_full).convert("RGB")

            processed += 1
            if args.progress and (
                processed == 1
                or processed % max(1, int(args.log_every)) == 0
                or processed == total
            ):
                pid = p.get("panel_id") if isinstance(p.get("panel_id"), str) else ""
                print(f"[Ollama] ({processed}/{total}) panel_id={pid} crop={crop_rel}")
                sys.stdout.flush()

            t0 = time.time()
            try:
                # Caption prompt now includes character hints — no hardcoding
                caption = demo._ollama_generate_text(
                    host=args.ollama_host,
                    model=args.ollama_model,
                    prompt=caption_prompt,
                    image=crop,
                    max_tokens=int(args.ollama_caption_tokens),
                    temperature=0.2,
                    timeout_s=max(5, int(args.ollama_timeout)),
                )
            except Exception as e:
                caption = f"ERROR: {e}"
            p["scene_caption"] = (caption or "").strip()

            if args.progress and (
                processed == 1
                or processed % max(1, int(args.log_every)) == 0
                or processed == total
            ):
                dt = time.time() - t0
                print(f"[Ollama] ({processed}/{total}) done in {dt:.1f}s → {p['scene_caption'][:80]}")
                sys.stdout.flush()

            try:
                tags_text = demo._ollama_generate_text(
                    host=args.ollama_host,
                    model=args.ollama_model,
                    prompt=demo._ollama_tags_prompt(p["scene_caption"]),
                    image=None,
                    max_tokens=int(args.ollama_tags_tokens),
                    temperature=0.0,
                    timeout_s=max(5, int(args.ollama_timeout)),
                )
                tags, _ = demo._parse_scene_labels(tags_text)
                p["scene_tags"] = tags
            except Exception:
                if not isinstance(p.get("scene_tags"), list):
                    p["scene_tags"] = []

        if args.progress and total:
            dt_all = time.time() - start_all
            print(f"[Ollama] done: {processed}/{total} panels in {dt_all / 60.0:.1f} min")
            sys.stdout.flush()

    # ── Save Gemini cache ─────────────────────────────────────────────────────
    if args.cache and gemini_rotator is not None:
        try:
            cache_path.write_text(
                json.dumps(demo._to_jsonable(gemini_cache), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── Write per-panel sidecars ──────────────────────────────────────────────
    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    for p in panels:
        if not isinstance(p, dict):
            continue
        crop_rel = p.get("crop_path")
        if not isinstance(crop_rel, str) or not crop_rel.strip():
            continue
        crop_path_full = out_root / crop_rel
        panel_dir = crop_path_full.parent
        if not panel_dir.exists():
            continue
        caption = (p.get("scene_caption") or "").strip() if isinstance(p.get("scene_caption"), str) else ""
        tags = p.get("scene_tags") if isinstance(p.get("scene_tags"), list) else []
        try:
            (panel_dir / "scene.json").write_text(
                json.dumps(
                    {
                        "scene_caption": caption,
                        "scene_tags": tags,
                        "provider": scene_provider,
                        "updated_at": now_utc,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (panel_dir / "scene.txt").write_text(
                caption + ("\n" if caption else ""),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── Write output storyboard ───────────────────────────────────────────────
    out_path = Path(args.out) if args.out else storyboard_path
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"- panels updated: {sum(1 for p in panels if isinstance(p, dict))}")


if __name__ == "__main__":
    main()