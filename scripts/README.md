# Pipeline (Two-Step)

## Stage 1: Magi-only extraction

Runs Magi panel detection + OCR, crops each detected panel (yellow boxes), and writes per-panel transcripts.

```bash
python scripts/extract_chapter.py \
  --chapter-id "series/ch001" \
  --images /path/to/pages/ \
  --out /path/to/out_ch001
```

Outputs (under `--out`):

- `final/storyboard.json` (scene fields empty)
- `final/pages/000/panels/000/panel.png`
- `final/pages/000/panels/000/transcript.txt`
- `final/pages/000/panels/000/transcript.json`
- `final/pages/000/panels/000/panel.json`

## Stage 2: Scene enrichment

Adds `scene_caption` + `scene_tags` to an existing `final/storyboard.json` and writes `scene.json`/`scene.txt` next to each cropped panel.

```bash
python scripts/add_scenes.py /path/to/out_ch001/final/storyboard.json --scene-provider auto --cache
```

## Stage 3: Recap script (Ollama)

Generates recap text using `panel.png` + `scene_caption/tags` + transcript, carrying forward prior recap for continuity.

```bash
python scripts/make_panel_recaps.py /path/to/out_ch001/final/storyboard.json \
  --ollama-model qwen2.5vl:7b \
  --progress
```

Outputs:

- Per-panel: `recap.txt` + `recap.json` next to each `panel.png`
- Chapter-level: `final/recap_script.txt` + `final/panel_recaps.jsonl`

### Optional: Page-level script instead of per-crop

Generate **one recap per page** using all cropped sub-panels under `final/pages/<page>/panels/*`:

```bash
python scripts/make_panel_recaps.py /path/to/out_ch001/final/storyboard.json \
  --mode page \
  --ollama-model qwen2.5vl:7b \
  --progress
```

Outputs:

- Chapter-level: `final/recap_pages.json` (includes `raw_script` plus per-page recap + panel linkage)

## Stage 4: Final coherent recap (Groq rewrite over chunks)

Takes the raw per-panel recaps from Stage 3, groups panels into chunks (3–5 recommended), and asks Groq to rewrite into a smooth, coherent recap while keeping continuity across chunks.

```bash
export GROQ_SCRIPT_KEY="..."
export GROQ_SCRIPT_MODEL="..."
python scripts/make_final_recap.py /path/to/out_ch001/final/storyboard.json --chunk-size 4 --progress
```

Outputs:

- `final/recap_final.txt`
- `final/recap_chunks.jsonl`
- `final/recap_chunks/c0000.txt` (+ `c0000.raw.json` prompts for debugging)

## Optional Stage 4b: Blog-style recap + paragraph→page mapping (Gemini)

Takes the Stage 3 page-mode output `final/recap_pages.json` and asks Gemini to produce a cohesive 3–5 paragraph blog recap,
plus a `segment_map` that links each paragraph to the page indices it primarily draws from.

```bash
export GEMINI_API_KEY="..."
python scripts/make_blog_recap.py /path/to/out_ch001/final/recap_pages.json \
  --provider gemini \
  --gemini-model gemini-2.5-flash \
  --progress
```

Outputs:

- `final/recap_blog.json`

## Stage 5 (Optional): TTS audio generation for recap JSONs (Kokoro)

If you have a recap JSON in the “Claude recap format” (page recap + per-panel sentences),
you can generate WAV audio and attach `audio_path` fields for downstream video editing:

```bash
python video-pipeline/audio-generation/generate_audio.py \
  --input video-pipeline/audio-generation/input \
  --output video-pipeline/audio-generation/output
```

### Using Ollama instead (local models)

```bash
python scripts/make_blog_recap.py /path/to/out_ch001/final/recap_pages.json \
  --provider ollama \
  --ollama-model qwen2.5:7b \
  --ollama-host http://127.0.0.1:11434 \
  --progress
```

### Using Groq instead

```bash
export GROQ_SCRIPT_KEY="..."
export GROQ_SCRIPT_MODEL="..."   # e.g. llama-3.1-70b-versatile
python scripts/make_blog_recap.py /path/to/out_ch001/final/recap_pages.json \
  --provider groq \
  --groq-max-tokens 1200 \
  --progress
```

## Utility: Infer character names from OCR

Best-effort speaker tag (`char_0`, `char_1`, …) → character name mapping by scanning all `transcript.txt/json` in the chapter.

```bash
python scripts/infer_character_names.py /path/to/out_ch001/final/storyboard.json
```

Outputs:

- `final/character_map.json`
- `final/character_map_report.json` (debug/evidence)
