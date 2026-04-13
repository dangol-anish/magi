# Audio generation (Kokoro TTS)

This folder takes the **Stage 3 recap JSON** (your Claude “recap + per-panel sentences” format) and generates TTS audio for:

- `pages[].recap` → `pages[].recap_audio_path`
- `pages[].panels[].sentence` → `pages[].panels[].audio_path`

It writes:

- A new JSON with the audio paths attached
- WAV files under an `audio/` folder next to that JSON

## Quick start

1) Put your recap JSON(s) in:

- `video-pipeline/audio-generation/input/`

2) Generate audio + updated JSON(s):

```bash
python video-pipeline/audio-generation/generate_audio.py
```

Outputs land in:

- `video-pipeline/audio-generation/output/<input_basename>/<input_basename>.with_audio.json`
- `video-pipeline/audio-generation/output/<input_basename>/audio/...`

## Multiple folders under input/

If you store chapters like:

- `video-pipeline/audio-generation/input/ch_1/recap.json`
- `video-pipeline/audio-generation/input/ch_2/recap.json`

Run:

```bash
python video-pipeline/audio-generation/generate_audio.py --recursive
```

Outputs will mirror the subfolder structure to avoid name collisions (e.g. `output/ch_1/recap/...`).

Write back into the input JSON(s) instead:

```bash
python video-pipeline/audio-generation/generate_audio.py --in-place
```

## Options

Generate only page recaps:

```bash
python video-pipeline/audio-generation/generate_audio.py --pages
```

Generate only panel sentences:

```bash
python video-pipeline/audio-generation/generate_audio.py --panels
```

Also create one fully stitched WAV:

```bash
python video-pipeline/audio-generation/generate_audio.py --stitch
```

Stitched output is written to `audio/stitched/full.wav` and linked in the JSON as `stitched_audio_path` (plus `stitched_segments` timing metadata).

Choose engine/voice:

```bash
python video-pipeline/audio-generation/generate_audio.py --engine pykokoro --voice af_bella --speed 1.0
```

Disable progress output:

```bash
python video-pipeline/audio-generation/generate_audio.py --no-progress
```

## Better error handling (retries + report)

By default, each audio segment retries a few times. You can tune it and continue even if some segments fail:

```bash
python video-pipeline/audio-generation/generate_audio.py \
  --retry 5 --retry-wait-ms 500 --retry-backoff 2.0 \
  --allow-failures
```

If failures happen, the output JSON includes `tts_failures` (with error type/message/traceback) and the CLI exits with code `2`.

## Dependencies

This script tries `pykokoro` first, then falls back to `kokoro`.

Install deps (recommended):

```bash
python -m pip install -r video-pipeline/audio-generation/requirements.txt
python -m spacy download en_core_web_sm
```

Notes:

- `pykokoro` uses spaCy for tokenization and (by default) expects `en_core_web_sm`.
- The first TTS run will download model files into `video-pipeline/audio-generation/.cache` (configurable via `--cache-dir`).
