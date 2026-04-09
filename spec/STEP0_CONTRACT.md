# Step 0: Storyboard Contract (v1)

This repo’s downstream goal is to create recap-style video timelines where audio segments are synchronized with manga panel visuals.

The **only required final artifact** for downstream rendering is a single JSON file plus a folder of panel crops (and later audio).

## Outputs (minimal default)

For a given chapter output root:

- `final/storyboard.json` — conforms to `spec/storyboard.schema.json` (`version: "v1"`).
- `final/panels/` — cropped panel PNGs referenced by `panels[].crop_path`.
- `final/audio/` — generated audio referenced by `script[].audio_path` (Stage 3+).

Optional debug outputs (only when explicitly enabled):

- annotated overlays, raw model dumps, caches, usage metadata, etc.

## Status semantics

Every `beat` and every `script` line has a `status`:

- `OK`: grounded and usable as-is
- `UNCERTAIN`: ambiguous but still produces a continuity line
- `BLOCKED`: model cannot determine; must still output a placeholder + a reason so nothing is silently skipped

## IDs (recommended)

IDs must be stable across re-runs.

- `panel_id`: `ch{chapter}_p{page}_n{index}_{hash10}`
  - `hash10` can be derived from a crop hash or a (page image hash + bbox) hash.
- `beat_id`: deterministic like `b000`, `b001`… within the chapter.
- `line_id`: deterministic like `l000`, `l001`… within the chapter.

