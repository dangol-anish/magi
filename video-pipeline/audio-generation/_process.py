from __future__ import annotations

import sys
import traceback
from pathlib import Path

from _audio_paths import audio_path_for
from _errors import fail, short_exc
from _json_io import load_json, write_json
from _progress import Progress
from _retry import synth_with_retry
from _stitch import stitch_segments
from _targets import iter_targets
from _text import normalize_text
from _time import now_utc_iso
from _tts import TtsEngine
from _wav_io import write_wav


def process_one(
    *,
    in_json: Path,
    out_root: Path,
    out_dir: Path | None,
    engine: TtsEngine,
    overwrite: bool,
    include_pages: bool,
    include_panels: bool,
    dry_run: bool,
    in_place: bool,
    only_page_idx: int | None,
    max_items: int | None,
    stitch: bool,
    stitch_gap_ms: float,
    progress: Progress | None,
    retry_attempts: int,
    retry_wait_s: float,
    retry_backoff: float,
    allow_failures: bool,
) -> Path:
    data = load_json(in_json)

    base = in_json.stem
    if in_place:
        chapter_out = in_json.parent
        out_json = in_json
    else:
        chapter_out = out_dir if out_dir is not None else (out_root / base)
        out_json = chapter_out / f"{base}.with_audio.json"

    meta = dict(engine.meta)
    meta.update(
        {
            "tts_generated_at": now_utc_iso(),
            "audio_root": "audio",
            "source_json": str(in_json),
        }
    )
    data.setdefault("tts", {})
    if isinstance(data.get("tts"), dict):
        data["tts"].update(meta)
    else:
        data["tts"] = meta

    failures: list[dict] = []
    stitched_segments: list[dict] = []

    processed_local = 0
    for kind, obj, page_idx, text in iter_targets(data, include_pages=include_pages, include_panels=include_panels):
        if only_page_idx is not None and page_idx is not None and page_idx != only_page_idx:
            continue
        if only_page_idx is not None and page_idx is None:
            continue

        text_n = normalize_text(text)
        if not text_n:
            continue

        panel_id = obj.get("panel_id") if isinstance(obj.get("panel_id"), str) else None
        rel_audio = audio_path_for(kind=kind, page_idx=page_idx, panel_id=panel_id)
        audio_key = "recap_audio_path" if kind == "page_recap" else "audio_path"
        out_audio = chapter_out / rel_audio

        if (not overwrite) and out_audio.exists():
            obj[audio_key] = rel_audio
            stitched_segments.append(
                {
                    "kind": kind,
                    "page_idx": page_idx,
                    "panel_id": panel_id,
                    "text": text_n,
                    "audio_path": rel_audio,
                    "status": "skipped",
                }
            )
            if progress is not None:
                progress.update(processed_delta=1, skipped_delta=1)
            processed_local += 1
            if max_items is not None and processed_local >= max_items:
                break
            continue

        if dry_run:
            obj[audio_key] = rel_audio
            stitched_segments.append(
                {
                    "kind": kind,
                    "page_idx": page_idx,
                    "panel_id": panel_id,
                    "text": text_n,
                    "audio_path": rel_audio,
                    "status": "dry_run",
                }
            )
            if progress is not None:
                progress.update(processed_delta=1, generated_delta=1)
            processed_local += 1
            if max_items is not None and processed_local >= max_items:
                break
            continue

        try:
            res = synth_with_retry(
                engine=engine,
                text=text_n,
                attempts=int(retry_attempts),
                base_wait_s=float(retry_wait_s),
                backoff=float(retry_backoff),
            )
            write_wav(out_audio, res.audio, res.sample_rate)
            obj[audio_key] = rel_audio
            stitched_segments.append(
                {
                    "kind": kind,
                    "page_idx": page_idx,
                    "panel_id": panel_id,
                    "text": text_n,
                    "audio_path": rel_audio,
                    "status": "generated",
                }
            )
            if progress is not None:
                progress.update(processed_delta=1, generated_delta=1)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            err_type = e.__class__.__name__
            err_msg = short_exc(e)
            print(
                f"FAIL: {kind} page_idx={page_idx} panel_id={panel_id or '-'} -> {rel_audio} | {err_type}: {err_msg}",
                file=sys.stderr,
                flush=True,
            )
            record = {
                "kind": kind,
                "page_idx": page_idx,
                "panel_id": panel_id,
                "text": text_n,
                "audio_path": rel_audio,
                "error_type": err_type,
                "error_message": err_msg,
                "traceback": traceback.format_exc(limit=8),
                "attempts": int(retry_attempts),
            }
            failures.append(record)
            if kind == "page_recap":
                obj["recap_audio_error"] = {"type": err_type, "message": err_msg, "attempts": int(retry_attempts)}
            else:
                obj["audio_error"] = {"type": err_type, "message": err_msg, "attempts": int(retry_attempts)}
            stitched_segments.append({**record, "status": "failed"})
            if progress is not None:
                progress.update(processed_delta=1, failed_delta=1)
            if not allow_failures:
                data["tts_failures"] = failures
                data["tts_failure_count"] = len(failures)
                write_json(out_json, data)
                fail(
                    f"TTS generation failed for {kind} (page_idx={page_idx}, panel_id={panel_id}): {err_type}: {err_msg}"
                )

        processed_local += 1
        if max_items is not None and processed_local >= max_items:
            break

    if stitch and not dry_run:
        stitched_rel = "audio/stitched/full.wav"
        stitched_meta = stitch_segments(
            chapter_out=chapter_out,
            stitched_segments=stitched_segments,
            stitched_rel=stitched_rel,
            overwrite=overwrite,
            gap_ms=stitch_gap_ms,
        )
        data.update(stitched_meta)

    if failures:
        data["tts_failures"] = failures
        data["tts_failure_count"] = len(failures)

    if not dry_run:
        write_json(out_json, data)
    return out_json
