#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a storyboard.json contract (lightweight checks).")
    ap.add_argument("path", help="Path to storyboard JSON (e.g. final/storyboard.json)")
    ap.add_argument(
        "--root",
        default=None,
        help="Optional chapter output root. If set, verifies referenced crop/audio paths exist under it.",
    )
    ap.add_argument(
        "--require-coverage",
        action="store_true",
        help="Require that every panel_id appears in exactly one beat, and every beat has >=1 script line.",
    )
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        _fail(f"File not found: {p}")

    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        _fail(f"Invalid JSON: {e}")

    if not isinstance(doc, dict):
        _fail("Top-level JSON must be an object.")

    for k in ("version", "chapter_id", "panels", "beats", "script"):
        if k not in doc:
            _fail(f"Missing top-level key: {k!r}")

    panels = doc.get("panels")
    if not isinstance(panels, list) or not panels:
        _fail("panels must be a non-empty array.")

    panel_ids: list[str] = []
    for i, panel in enumerate(panels):
        if not isinstance(panel, dict):
            _fail(f"panels[{i}] must be an object.")
        pid = panel.get("panel_id")
        if not isinstance(pid, str) or not pid.strip():
            _fail(f"panels[{i}].panel_id must be a non-empty string.")
        panel_ids.append(pid)

        for required in ("page_idx", "bbox", "crop_path", "ocr_lines", "scene_caption"):
            if required not in panel:
                _fail(f"panels[{i}] missing required key {required!r}")

        bbox = panel.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox)):
            _fail(f"panels[{i}].bbox must be [x1,y1,x2,y2] numbers.")

    dup = {x for x in panel_ids if panel_ids.count(x) > 1}
    if dup:
        _fail(f"Duplicate panel_id(s): {sorted(dup)}")

    beats = doc.get("beats")
    if not isinstance(beats, list):
        _fail("beats must be an array.")

    beat_ids: list[str] = []
    beat_to_panels: dict[str, list[str]] = {}
    for i, beat in enumerate(beats):
        if not isinstance(beat, dict):
            _fail(f"beats[{i}] must be an object.")
        bid = beat.get("beat_id")
        if not isinstance(bid, str) or not bid.strip():
            _fail(f"beats[{i}].beat_id must be a non-empty string.")
        beat_ids.append(bid)
        pids = beat.get("panel_ids")
        if not (isinstance(pids, list) and pids and all(isinstance(x, str) and x.strip() for x in pids)):
            _fail(f"beats[{i}].panel_ids must be a non-empty array of strings.")
        beat_to_panels[bid] = pids
        if beat.get("status") not in {"OK", "UNCERTAIN", "BLOCKED"}:
            _fail(f"beats[{i}].status must be OK|UNCERTAIN|BLOCKED.")

    script = doc.get("script")
    if not isinstance(script, list):
        _fail("script must be an array.")

    beat_to_lines: dict[str, int] = {}
    for i, line in enumerate(script):
        if not isinstance(line, dict):
            _fail(f"script[{i}] must be an object.")
        if line.get("status") not in {"OK", "UNCERTAIN", "BLOCKED"}:
            _fail(f"script[{i}].status must be OK|UNCERTAIN|BLOCKED.")
        bid = line.get("beat_id")
        if not isinstance(bid, str) or not bid.strip():
            _fail(f"script[{i}].beat_id must be a non-empty string.")
        beat_to_lines[bid] = beat_to_lines.get(bid, 0) + 1
        pids = line.get("panel_ids")
        if not (isinstance(pids, list) and pids and all(isinstance(x, str) and x.strip() for x in pids)):
            _fail(f"script[{i}].panel_ids must be a non-empty array of strings.")

    if args.require_coverage:
        # Every panel appears in exactly one beat.
        seen: dict[str, int] = {}
        for bid, pids in beat_to_panels.items():
            for pid in pids:
                seen[pid] = seen.get(pid, 0) + 1
        missing = [pid for pid in panel_ids if seen.get(pid, 0) == 0]
        multi = [pid for pid, c in seen.items() if c > 1]
        if missing:
            _fail(f"Panels missing from beats: {missing[:20]}" + (" (more...)" if len(missing) > 20 else ""))
        if multi:
            _fail(f"Panels appear in multiple beats: {multi[:20]}" + (" (more...)" if len(multi) > 20 else ""))
        # Every beat must have >= 1 script line.
        no_lines = [bid for bid in beat_ids if beat_to_lines.get(bid, 0) == 0]
        if no_lines:
            _fail(f"Beats with no script lines: {no_lines[:20]}" + (" (more...)" if len(no_lines) > 20 else ""))

    if args.root:
        root = Path(args.root)
        if not root.exists():
            _fail(f"--root does not exist: {root}")
        for panel in panels:
            crop = panel.get("crop_path")
            if isinstance(crop, str) and crop.strip():
                fp = root / crop
                if not fp.exists():
                    _fail(f"Missing crop file: {fp}")
        for line in script:
            audio = line.get("audio_path")
            if isinstance(audio, str) and audio.strip():
                fp = root / audio
                if not fp.exists():
                    _fail(f"Missing audio file: {fp}")

    print("OK: storyboard contract looks sane.")


if __name__ == "__main__":
    main()

