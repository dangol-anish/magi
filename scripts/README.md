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
