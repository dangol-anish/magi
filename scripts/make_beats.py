#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1.5: convert panels[] into beats[] (deterministic baseline).")
    ap.add_argument("storyboard", help="Path to final/storyboard.json")
    ap.add_argument("--in-place", action="store_true", help="Overwrite the input file (default).")
    ap.add_argument("--out", default=None, help="Write to a different path instead of overwriting.")
    ap.add_argument("--overwrite", action="store_true", help="Replace existing beats[].")
    ap.add_argument(
        "--mode",
        choices=["one-per-panel"],
        default="one-per-panel",
        help="Beat segmentation strategy (start simple).",
    )
    args = ap.parse_args()

    in_path = Path(args.storyboard)
    if not in_path.exists():
        _fail(f"File not found: {in_path}")

    doc = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        _fail("Top-level storyboard must be a JSON object.")

    panels = doc.get("panels")
    if not isinstance(panels, list) or not panels:
        _fail("storyboard.panels must be a non-empty array.")

    if isinstance(doc.get("beats"), list) and doc.get("beats") and not args.overwrite:
        print("Beats already exist; pass --overwrite to replace them.")
        return

    beats = []
    if args.mode == "one-per-panel":
        for idx, p in enumerate(panels):
            if not isinstance(p, dict):
                _fail(f"panels[{idx}] must be an object.")
            panel_id = p.get("panel_id")
            if not isinstance(panel_id, str) or not panel_id.strip():
                _fail(f"panels[{idx}].panel_id must be a non-empty string.")
            scene_caption = (p.get("scene_caption") or "").strip() if isinstance(p.get("scene_caption"), str) else ""
            ocr_lines = p.get("ocr_lines") if isinstance(p.get("ocr_lines"), list) else []
            has_ocr = any(isinstance(x, dict) and isinstance(x.get("text"), str) and x.get("text").strip() for x in ocr_lines)

            status = "OK" if (scene_caption or has_ocr) else "UNCERTAIN"
            summary = scene_caption if scene_caption else ""
            beats.append(
                {
                    "beat_id": f"b{idx:04d}",
                    "panel_ids": [panel_id],
                    "summary": summary,
                    "status": status,
                }
            )

    doc["beats"] = beats

    out_path = Path(args.out) if args.out else in_path
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"- beats: {len(beats)}")


if __name__ == "__main__":
    main()

