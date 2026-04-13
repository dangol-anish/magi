from __future__ import annotations

from pathlib import Path

import numpy as np

from _errors import fail
from _wav_io import read_wav, samples_to_ms, silence, write_wav


def stitch_segments(
    *,
    chapter_out: Path,
    stitched_segments: list[dict],
    stitched_rel: str,
    overwrite: bool,
    gap_ms: float,
) -> dict:
    stitched_abs = chapter_out / stitched_rel
    if (not overwrite) and stitched_abs.exists():
        return {"stitched_audio_path": stitched_rel}

    sr0: int | None = None
    parts: list[np.ndarray] = []
    starts_ends: list[tuple[int, int] | None] = []
    cursor = 0
    gap_audio: np.ndarray | None = None

    for seg in stitched_segments:
        wav_rel = seg.get("audio_path")
        if not isinstance(wav_rel, str) or not wav_rel.strip():
            starts_ends.append(None)
            continue
        wav_abs = chapter_out / wav_rel
        if not wav_abs.exists():
            starts_ends.append(None)
            continue

        audio, sr = read_wav(wav_abs)
        if sr0 is None:
            sr0 = sr
            gap_audio = silence(sr0, gap_ms)
        elif sr != sr0:
            fail(f"Sample rate mismatch while stitching ({sr} != {sr0}): {wav_abs}")

        if parts and gap_audio is not None and gap_audio.size:
            parts.append(gap_audio)
            cursor += int(gap_audio.size)
        start = cursor
        parts.append(audio)
        cursor += int(audio.size)
        end = cursor
        starts_ends.append((start, end))

    if sr0 is None:
        fail("Nothing to stitch (no segments produced).")

    stitched_audio = np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)
    write_wav(stitched_abs, stitched_audio, sr0)

    seg_out: list[dict] = []
    for seg, se in zip(stitched_segments, starts_ends, strict=False):
        if se is None:
            seg_out.append({**seg, "start_ms": None, "end_ms": None})
        else:
            s, e = se
            seg_out.append({**seg, "start_ms": samples_to_ms(s, sr0), "end_ms": samples_to_ms(e, sr0)})

    return {
        "stitched_audio_path": stitched_rel,
        "stitched_sample_rate": sr0,
        "stitched_gap_ms": float(gap_ms),
        "stitched_segments": seg_out,
    }

