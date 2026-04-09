#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


def _fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])


def _looks_like_quote(text: str) -> bool:
    # Heuristic: disallow obvious quoting punctuation.
    t = text or ""
    return any(ch in t for ch in ['"', "“", "”", "’", "‘"])


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _offline_recap_for_beat(*, beat_panel_ids: list[str], scene_caps: list[str]) -> list[dict]:
    """
    Deterministic placeholder: 2 short narrator sentences, no quotes.
    Used for testing the pipeline without an LLM call.
    """
    cap = ""
    for c in scene_caps:
        c = (c or "").strip()
        if c:
            cap = c
            break

    if cap:
        s1 = cap
    else:
        s1 = "The scene continues, but details are unclear."
    s1 = " ".join(s1.split())
    if _looks_like_quote(s1):
        s1 = s1.replace('"', "").replace("“", "").replace("”", "")

    # Keep short.
    words = s1.split()
    if len(words) > 12:
        s1 = " ".join(words[:12]).rstrip(".") + "."

    s2 = "The moment sets up what happens next."
    # Distribute panels across sentences for fast cuts.
    p1 = beat_panel_ids[:1]
    p2 = beat_panel_ids[1:2] or p1
    return [{"text": s1, "panel_ids": p1}, {"text": s2, "panel_ids": p2}]


def _text_to_items(text: str, *, beat_panel_ids: list[str], max_items: int) -> list[dict]:
    """
    Best-effort converter when a provider doesn't return JSON.
    Splits into short sentences and assigns panels in reading order.
    """
    t = " ".join((text or "").strip().split())
    if not t:
        return []
    t = t.replace('"', "").replace("“", "").replace("”", "").replace("’", "'")
    # Split on sentence boundaries / newlines.
    raw = re.split(r"(?<=[.!?])\s+|\n+", t)
    sents = [s.strip() for s in raw if s and s.strip()]
    sents = sents[: max(1, int(max_items))]
    out: list[dict] = []
    for i, s in enumerate(sents):
        pid = beat_panel_ids[min(i, len(beat_panel_ids) - 1) : min(i, len(beat_panel_ids) - 1) + 1]
        out.append({"text": s, "panel_ids": pid})
    return out


def _build_evidence(panel: dict, *, max_ocr_lines: int) -> tuple[list[str], list[str], list[str]]:
    scene_caps: list[str] = []
    scene_tags: list[str] = []
    ocr_lines: list[str] = []

    cap = panel.get("scene_caption")
    if isinstance(cap, str) and cap.strip():
        scene_caps.append(cap.strip())

    tags = panel.get("scene_tags")
    if isinstance(tags, list):
        for t in tags:
            if isinstance(t, str) and t.strip():
                scene_tags.append(t.strip().lower())

    ocr = panel.get("ocr_lines")
    if isinstance(ocr, list):
        for ln in ocr:
            if not isinstance(ln, dict):
                continue
            t = ln.get("text")
            if isinstance(t, str) and t.strip():
                ocr_lines.append(" ".join(t.strip().split()))

    if max_ocr_lines > 0:
        ocr_lines = ocr_lines[:max_ocr_lines]

    return scene_caps, scene_tags, ocr_lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 2 (recap): generate YouTube-style narrator script lines per beat, linked to panel_ids."
    )
    ap.add_argument("storyboard", help="Path to final/storyboard.json")
    ap.add_argument("--out", default=None, help="Write to a different path (default: overwrite input).")
    ap.add_argument("--overwrite", action="store_true", help="Replace existing script[].")

    ap.add_argument("--provider", choices=["none", "gemini", "ollama", "auto"], default="none")
    ap.add_argument("--sentences-min", type=int, default=2)
    ap.add_argument("--sentences-max", type=int, default=4)
    ap.add_argument("--max-words", type=int, default=12)
    ap.add_argument("--max-panels-per-sentence", type=int, default=2)
    ap.add_argument("--max-ocr-lines", type=int, default=6, help="Max OCR lines to include as evidence per panel.")

    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    ap.add_argument("--gemini-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--gemini-log-key", action="store_true")
    ap.add_argument("--gemini-timeout", type=int, default=120)
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", default="llava-phi3:latest")
    ap.add_argument("--ollama-timeout", type=int, default=120)
    args = ap.parse_args()

    min_s = _clamp_int(args.sentences_min, 1, 8)
    max_s = _clamp_int(args.sentences_max, min_s, 12)
    max_words = _clamp_int(args.max_words, 4, 30)
    max_panels = _clamp_int(args.max_panels_per_sentence, 1, 6)
    max_ocr = _clamp_int(args.max_ocr_lines, 0, 50)

    in_path = Path(args.storyboard)
    if not in_path.exists():
        _fail(f"File not found: {in_path}")

    doc = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        _fail("Top-level storyboard must be a JSON object.")

    panels = doc.get("panels")
    beats = doc.get("beats")
    if not isinstance(panels, list) or not panels:
        _fail("storyboard.panels must be a non-empty array.")
    if not isinstance(beats, list) or not beats:
        _fail("storyboard.beats must be a non-empty array. Run scripts/make_beats.py first.")

    if isinstance(doc.get("script"), list) and doc.get("script") and not args.overwrite:
        print("script[] already exists; pass --overwrite to replace it.")
        return

    panel_by_id = {}
    for p in panels:
        if isinstance(p, dict) and isinstance(p.get("panel_id"), str):
            panel_by_id[p["panel_id"]] = p

    # Optional provider wiring via existing demo utilities.
    rotator = None
    demo = None
    requests = None
    if args.provider in {"gemini", "auto", "ollama"}:
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        import requests as _requests  # type: ignore
        import examples.magiv3_demo as _demo  # type: ignore

        requests = _requests
        demo = _demo
        if args.provider in {"gemini", "auto"}:
            keys = demo._load_gemini_keys(args.gemini_key_env)
            if not keys:
                _fail(f"No Gemini keys found in env var {args.gemini_key_env!r} (+ _2.._5).")
            rotator = demo.GeminiKeyRotator(keys, log_key_fingerprint=bool(args.gemini_log_key))

    script_lines: list[dict] = []
    line_idx = 0

    for beat in beats:
        if not isinstance(beat, dict):
            continue
        beat_id = beat.get("beat_id")
        beat_panel_ids = beat.get("panel_ids")
        if not isinstance(beat_id, str) or not isinstance(beat_panel_ids, list) or not beat_panel_ids:
            continue

        # Gather text-only evidence for this beat.
        scene_caps_all: list[str] = []
        scene_tags_all: list[str] = []
        ocr_all: list[str] = []
        for pid in beat_panel_ids:
            panel = panel_by_id.get(pid)
            if not isinstance(panel, dict):
                continue
            caps, tags, ocr = _build_evidence(panel, max_ocr_lines=max_ocr)
            scene_caps_all.extend(caps)
            scene_tags_all.extend(tags)
            ocr_all.extend(ocr)

        # De-dup while preserving order
        def _dedup(xs: list[str]) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for x in xs:
                x = x.strip()
                if not x or x in seen:
                    continue
                out.append(x)
                seen.add(x)
            return out

        scene_caps_all = _dedup(scene_caps_all)[:6]
        scene_tags_all = _dedup(scene_tags_all)[:16]
        ocr_all = _dedup(ocr_all)[:24]

        # Persist compact evidence for debugging/audit in beat.notes (schema-safe).
        beat["notes"] = json.dumps(
            {"scene_captions": scene_caps_all, "scene_tags": scene_tags_all, "ocr_lines": ocr_all},
            ensure_ascii=False,
        )

        if args.provider == "none":
            recaps = _offline_recap_for_beat(beat_panel_ids=beat_panel_ids, scene_caps=scene_caps_all)
        else:
            assert demo is not None and requests is not None

            prompt = (
                "Write YouTube-style manga recap narration.\n"
                "- NEVER quote dialogue.\n"
                "- Do NOT use quotation marks.\n"
                f"- Write {min_s} to {max_s} short sentences.\n"
                f"- Each sentence must be <= {max_words} words.\n"
                "- Narrate as a story, paraphrasing what is conveyed.\n"
                "- Use ONLY the evidence provided.\n"
                "- Return ONLY valid JSON array, each item:\n"
                "  {\"text\":\"...\",\"panel_ids\":[\"...\"]}\n"
                f"- panel_ids in each item must be a subset of this beat's panel_ids.\n"
                f"- Each item should include 1 to {max_panels} panel_ids.\n\n"
                f"Beat panel_ids: {beat_panel_ids}\n\n"
                "Evidence (do not quote directly; paraphrase):\n"
                f"- Scene captions:\n  - " + "\n  - ".join(scene_caps_all) + "\n"
                f"- Scene tags:\n  - " + "\n  - ".join(scene_tags_all) + "\n"
                f"- OCR lines:\n  - " + "\n  - ".join(ocr_all) + "\n"
            )

            def _gemini_call():
                assert rotator is not None

                def _call(api_key: str):
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{args.gemini_model}:generateContent"
                    payload = {
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
                    }
                    r = requests.post(
                        url, params={"key": api_key}, json=payload, timeout=max(5, int(args.gemini_timeout))
                    )
                    r.raise_for_status()
                    raw = r.json()
                    text = ""
                    try:
                        parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        text = "".join([p.get("text", "") for p in parts if isinstance(p, dict)]).strip()
                    except Exception:
                        text = ""
                    arr_text = demo._extract_first_json_array(text)
                    if arr_text:
                        return json.loads(arr_text)
                    return _text_to_items(text, beat_panel_ids=beat_panel_ids, max_items=max_s)

                return rotator.call(_call)

            def _ollama_call():
                payload = {
                    "model": args.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 512},
                }
                r = requests.post(
                    f"{args.ollama_host.rstrip('/')}/api/generate",
                    json=payload,
                    timeout=max(5, int(args.ollama_timeout)),
                )
                r.raise_for_status()
                data = r.json()
                text = (data.get("response") or "").strip()
                arr_text = demo._extract_first_json_array(text)
                if arr_text:
                    return json.loads(arr_text)
                return _text_to_items(text, beat_panel_ids=beat_panel_ids, max_items=max_s)

            # Provider selection: gemini -> ollama -> offline (never skip).
            recaps = None
            err = None
            if args.provider in {"gemini", "auto"}:
                try:
                    recaps = _gemini_call()
                except Exception as e:
                    err = e
                    recaps = None
            if recaps is None and args.provider in {"ollama", "auto"}:
                try:
                    recaps = _ollama_call()
                except Exception as e:
                    err = e
                    recaps = None
            if recaps is None:
                recaps = [{"text": f"BLOCKED: {type(err).__name__ if err else 'unknown error'}", "panel_ids": beat_panel_ids[:1]}]

        # Validate + coerce recap items and emit script lines
        items_out: list[dict] = []
        if not isinstance(recaps, list):
            recaps = []
        for it in recaps:
            if not isinstance(it, dict):
                continue
            text = it.get("text")
            pids = it.get("panel_ids")
            if not isinstance(text, str):
                continue
            text = " ".join(text.strip().split())
            if not text:
                continue
            if _looks_like_quote(text):
                text = text.replace('"', "").replace("“", "").replace("”", "").strip()
            if _word_count(text) > max_words:
                text = " ".join(text.split()[:max_words]).rstrip(".") + "."
            if not (isinstance(pids, list) and pids):
                continue
            pids = [x for x in pids if isinstance(x, str) and x in beat_panel_ids]
            if not pids:
                continue
            pids = pids[:max_panels]
            items_out.append({"text": text, "panel_ids": pids})

        # Ensure we never skip: fallback to offline if provider output unusable.
        if len(items_out) < min_s:
            items_out = _offline_recap_for_beat(beat_panel_ids=beat_panel_ids, scene_caps=scene_caps_all)[:max_s]

        # Clamp to sentences_max
        items_out = items_out[:max_s]

        for it in items_out:
            text = it["text"]
            pids = it["panel_ids"]
            status = "OK"
            if text.startswith("BLOCKED:") or "unclear" in text.lower():
                status = "UNCERTAIN"
            script_lines.append(
                {
                    "line_id": f"l{line_idx:04d}",
                    "beat_id": beat_id,
                    "panel_ids": pids,
                    "text": text,
                    "status": status,
                }
            )
            line_idx += 1

    doc["beats"] = beats
    doc["script"] = script_lines

    out_path = Path(args.out) if args.out else in_path
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"- recap lines: {len(script_lines)}")


if __name__ == "__main__":
    main()
