#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _short_fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:10]


def _pick_panel_text(panel: dict) -> tuple[str, str]:
    """
    Returns (status, text) for a single-panel beat in offline mode.
    - Prefer OCR if available (dialogue-driven), else scene_caption.
    - Never return empty text; fall back to BLOCKED placeholder.
    """
    ocr_lines = panel.get("ocr_lines") if isinstance(panel.get("ocr_lines"), list) else []
    ocr_texts = []
    for x in ocr_lines:
        if isinstance(x, dict) and isinstance(x.get("text"), str) and x.get("text").strip():
            ocr_texts.append(x["text"].strip())

    if ocr_texts:
        # Keep short; we just need something testable and grounded.
        joined = " ".join(ocr_texts)
        joined = " ".join(joined.split())
        if len(joined) > 240:
            joined = joined[:240].rstrip() + "…"
        return "OK", joined

    cap = (panel.get("scene_caption") or "").strip() if isinstance(panel.get("scene_caption"), str) else ""
    if cap:
        return "OK", cap

    return "BLOCKED", "BLOCKED: no OCR or scene caption available for this panel."


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: generate line-level script[] entries linked to panel_ids.")
    ap.add_argument("storyboard", help="Path to final/storyboard.json")
    ap.add_argument("--out", default=None, help="Write to a different path (default: overwrite input).")
    ap.add_argument("--overwrite", action="store_true", help="Replace existing script[].")

    ap.add_argument("--provider", choices=["none", "gemini"], default="none")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    ap.add_argument("--gemini-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--gemini-log-key", action="store_true")
    ap.add_argument("--gemini-timeout", type=int, default=120)
    args = ap.parse_args()

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

    script_lines: list[dict] = []

    if args.provider == "none":
        # Deterministic baseline: 1 script line per beat.
        for idx, beat in enumerate(beats):
            if not isinstance(beat, dict):
                _fail(f"beats[{idx}] must be an object.")
            beat_id = beat.get("beat_id")
            panel_ids = beat.get("panel_ids")
            if not isinstance(beat_id, str) or not beat_id.strip():
                _fail(f"beats[{idx}].beat_id must be a non-empty string.")
            if not (isinstance(panel_ids, list) and panel_ids and all(isinstance(x, str) and x.strip() for x in panel_ids)):
                _fail(f"beats[{idx}].panel_ids must be a non-empty list of strings.")

            # For now we only support one-per-panel beats deterministically.
            pid = panel_ids[0]
            panel = panel_by_id.get(pid)
            if not isinstance(panel, dict):
                status, text = "BLOCKED", f"BLOCKED: missing panel for panel_id={pid!r}"
            else:
                status, text = _pick_panel_text(panel)

            line = {
                "line_id": f"l{idx:04d}",
                "beat_id": beat_id,
                "panel_ids": panel_ids,
                "text": text,
                "status": status,
            }
            if status != "OK":
                line["notes"] = "Offline baseline; fill via Gemini in provider=gemini mode."
            script_lines.append(line)

    else:
        # Gemini mode: create grounded narration per beat from OCR+scene info.
        # This is optional and can be introduced later; keep an explicit network dependency.
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        import requests  # type: ignore
        import examples.magiv3_demo as demo  # type: ignore

        keys = demo._load_gemini_keys(args.gemini_key_env)
        if not keys:
            _fail(f"No Gemini keys found in env var {args.gemini_key_env!r} (+ _2.._5).")
        if args.gemini_log_key:
            print(f"[Gemini] Key env base name: {args.gemini_key_env}")
            print(f"[Gemini] Key fingerprints (index → fingerprint): {list(enumerate([_short_fingerprint(k) for k in keys]))}")
        rotator = demo.GeminiKeyRotator(keys, log_key_fingerprint=args.gemini_log_key)

        def _gemini_text(api_key: str, prompt: str) -> str:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{args.gemini_model}:generateContent"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 256},
            }
            r = requests.post(url, params={"key": api_key}, json=payload, timeout=max(5, int(args.gemini_timeout)))
            r.raise_for_status()
            data = r.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join([p.get("text", "") for p in parts if isinstance(p, dict)]).strip()
            return text

        for idx, beat in enumerate(beats):
            beat_id = beat.get("beat_id")
            panel_ids = beat.get("panel_ids") if isinstance(beat.get("panel_ids"), list) else []
            panels_for_beat = [panel_by_id.get(pid) for pid in panel_ids if pid in panel_by_id]
            ocr = []
            caps = []
            for p in panels_for_beat:
                if not isinstance(p, dict):
                    continue
                caps.append((p.get("scene_caption") or "").strip() if isinstance(p.get("scene_caption"), str) else "")
                for ln in (p.get("ocr_lines") or []):
                    if isinstance(ln, dict) and isinstance(ln.get("text"), str) and ln.get("text").strip():
                        ocr.append(ln["text"].strip())

            prompt = (
                "You are writing a recap narrator line for a manga panel/beat.\n"
                "Use ONLY the provided OCR quotes and scene captions. Do not invent details.\n"
                "Write ONE short sentence (<= 25 words).\n\n"
                f"OCR quotes:\n- " + "\n- ".join(ocr[:12]) + "\n\n"
                f"Scene captions:\n- " + "\n- ".join([c for c in caps if c][:8]) + "\n"
            )

            try:
                text = rotator.call(lambda k: _gemini_text(k, prompt))
                text = (text or "").strip()
            except requests.exceptions.RequestException as e:
                _fail(
                    "Gemini request failed (network/transport error). "
                    "Re-run with network access enabled. "
                    f"error_type={type(e).__name__}"
                )
            except Exception as e:
                text = f"BLOCKED: {e}"

            status = "OK" if text and not text.startswith("BLOCKED:") else "BLOCKED"
            script_lines.append(
                {
                    "line_id": f"l{idx:04d}",
                    "beat_id": beat_id,
                    "panel_ids": panel_ids if panel_ids else [],
                    "text": text if text else "BLOCKED: empty response",
                    "status": status,
                }
            )

    doc["script"] = script_lines

    out_path = Path(args.out) if args.out else in_path
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"- script lines: {len(script_lines)}")


if __name__ == "__main__":
    main()

