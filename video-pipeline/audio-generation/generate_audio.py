#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from _errors import fail
from _json_io import load_json
from _process import process_one
from _progress import Progress
from _targets import count_targets
from _tts import DryRunEngine, pick_engine


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Kokoro TTS audio for recap JSONs and attach audio paths.")
    ap.add_argument(
        "--input",
        default=str(Path("video-pipeline/audio-generation/input")),
        help="Input JSON file or directory (default: video-pipeline/audio-generation/input)",
    )
    ap.add_argument(
        "--output",
        default=str(Path("video-pipeline/audio-generation/output")),
        help="Output directory (default: video-pipeline/audio-generation/output)",
    )
    ap.add_argument("--engine", default="auto", help="auto | pykokoro | kokoro (default: auto)")
    ap.add_argument("--voice", default="af_bella", help="Voice id (default: af_bella)")
    ap.add_argument("--speed", type=float, default=1.0, help="Speech speed (default: 1.0)")
    ap.add_argument("--kokoro-lang", default="a", help="kokoro KPipeline lang_code (default: a)")
    ap.add_argument(
        "--cache-dir",
        default=str(Path("video-pipeline/audio-generation/.cache")),
        help="Cache directory for model downloads (default: video-pipeline/audio-generation/.cache)",
    )
    ap.add_argument("--pages", action="store_true", help="Generate audio for pages[].recap")
    ap.add_argument("--panels", action="store_true", help="Generate audio for pages[].panels[].sentence")
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="When --input is a directory, also process *.json in subfolders (e.g. input/ch_1/recap.json)",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing audio files")
    ap.add_argument("--dry-run", action="store_true", help="Do not run TTS or write files; just show what would happen")
    ap.add_argument("--in-place", action="store_true", help="Write updated JSON back to the input file(s)")
    ap.add_argument("--page-idx", type=int, default=None, help="Only process a single page_idx (for quick tests)")
    ap.add_argument("--max-items", type=int, default=None, help="Stop after processing N items (for quick tests)")
    ap.add_argument("--stitch", action="store_true", help="Also stitch all generated segments into one WAV")
    ap.add_argument("--stitch-gap-ms", type=float, default=250.0, help="Silence gap between stitched segments (ms)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress output")
    ap.add_argument("--retry", type=int, default=3, help="Retry attempts for each segment (default: 3)")
    ap.add_argument("--retry-wait-ms", type=float, default=500.0, help="Initial retry wait (ms) (default: 500)")
    ap.add_argument("--retry-backoff", type=float, default=2.0, help="Retry backoff multiplier (default: 2.0)")
    ap.add_argument(
        "--allow-failures",
        action="store_true",
        help="Continue on failed segments and write a failure report into the output JSON",
    )

    args = ap.parse_args()

    input_path = Path(args.input).expanduser()
    out_root = Path(args.output).expanduser()

    include_pages = bool(args.pages)
    include_panels = bool(args.panels)
    if not include_pages and not include_panels:
        include_pages = True
        include_panels = True

    if not input_path.exists():
        fail(f"Input path does not exist: {input_path}")

    if args.dry_run:
        engine = DryRunEngine()
    else:
        cache_dir = str(Path(args.cache_dir).expanduser())
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        engine = pick_engine(
            voice=args.voice,
            speed=float(args.speed),
            prefer=args.engine,
            lang_code=args.kokoro_lang,
            cache_dir=cache_dir,
        )

    json_paths: list[Path] = []
    if input_path.is_dir():
        if bool(args.recursive):
            json_paths = sorted([p for p in input_path.rglob("*.json") if p.is_file()])
        else:
            json_paths = sorted([p for p in input_path.glob("*.json") if p.is_file()])
    else:
        if input_path.suffix.lower() != ".json":
            fail(f"Expected a .json file: {input_path}")
        json_paths = [input_path]

    if not json_paths:
        fail(f"No .json files found under: {input_path}")

    if shutil.which("espeak-ng") is None and args.engine in {"auto", "kokoro"}:
        print("WARN: espeak-ng not found in PATH; Kokoro engines may fail to phonemize text.", file=sys.stderr)

    if not bool(args.in_place):
        out_root.mkdir(parents=True, exist_ok=True)

    progress: Progress | None = None
    if not bool(args.no_progress):
        try:
            total = 0
            for p in json_paths:
                total += count_targets(
                    load_json(p),
                    include_pages=include_pages,
                    include_panels=include_panels,
                    only_page_idx=args.page_idx,
                )
            progress = Progress(total=total, enabled=True)
            progress.update()
        except Exception:
            progress = Progress(total=0, enabled=True)
            progress.update()

    failure_count_total = 0
    failure_types: dict[str, int] = {}

    for p in json_paths:
        out_dir: Path | None = None
        if not bool(args.in_place):
            # Preserve old layout (output/<basename>/...) for root-level inputs.
            # For recursive inputs, mirror subfolder structure to avoid name collisions.
            try:
                rel_parent = p.parent.relative_to(input_root)
            except Exception:
                rel_parent = Path()
            if bool(args.recursive) and str(rel_parent) not in {"", "."}:
                out_dir = out_root / rel_parent / p.stem
            else:
                out_dir = out_root / p.stem

        out_json = process_one(
            in_json=p,
            out_root=out_root,
            out_dir=out_dir,
            engine=engine,
            overwrite=bool(args.overwrite),
            include_pages=include_pages,
            include_panels=include_panels,
            dry_run=bool(args.dry_run),
            in_place=bool(args.in_place),
            only_page_idx=args.page_idx,
            max_items=(int(args.max_items) if args.max_items is not None and int(args.max_items) > 0 else None),
            stitch=bool(args.stitch),
            stitch_gap_ms=float(args.stitch_gap_ms),
            progress=progress,
            retry_attempts=int(args.retry),
            retry_wait_s=float(args.retry_wait_ms) / 1000.0,
            retry_backoff=float(args.retry_backoff),
            allow_failures=bool(args.allow_failures),
        )
        try:
            dd = load_json(Path(out_json))
            failures = dd.get("tts_failures")
            if isinstance(failures, list):
                for f in failures:
                    if not isinstance(f, dict):
                        continue
                    t = f.get("error_type")
                    if isinstance(t, str) and t:
                        failure_types[t] = int(failure_types.get(t, 0)) + 1
                failure_count_total += int(dd.get("tts_failure_count") or 0)
        except Exception:
            pass

        print(str(out_json))

    if progress is not None:
        progress.done()

    if failure_count_total and bool(args.allow_failures):
        types_s = (
            ", ".join([f"{k}={v}" for k, v in sorted(failure_types.items(), key=lambda kv: (-kv[1], kv[0]))])
            if failure_types
            else "(unknown)"
        )
        print(
            f"Done with failures: {failure_count_total} segment(s) failed. Types: {types_s}. See tts_failures in the output JSON.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
    input_root = input_path if input_path.is_dir() else input_path.parent
