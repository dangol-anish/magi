#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _maybe_load_dotenv() -> None:
    """
    Best-effort `.env` loader from project root.
    Does not override existing environment variables.
    """
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"
    if not dotenv_path.exists():
        return
    try:
        content = dotenv_path.read_text(encoding="utf-8")
    except Exception:
        return
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


_CROP_PATH_RE = re.compile(r"(?:^|/)final/pages/(\d+)/panels/(\d+)/")
_SPEAKER_TAG_RE = re.compile(r"^<([^>]+)>:\s*(.*)\s*$")
_BY_NAME_RE = re.compile(r"\bby\s+([A-Z][a-z]+)(?:\s+[A-Z][a-z]+)?\b")
_TITLED_NAME_RE = re.compile(r"\b(Mr\.|Ms\.|Mrs\.|Dr\.)\s+([A-Z][a-z]+)\b")


def _panel_sort_key(panel: dict, fallback_idx: int) -> tuple[int, int, int]:
    crop_path = panel.get("crop_path")
    if isinstance(crop_path, str):
        m = _CROP_PATH_RE.search(crop_path.replace("\\", "/"))
        if m:
            try:
                return (int(m.group(1)), int(m.group(2)), fallback_idx)
            except Exception:
                pass
    page_idx = panel.get("page_idx")
    if isinstance(page_idx, int):
        return (int(page_idx), fallback_idx, fallback_idx)
    return (fallback_idx, fallback_idx, fallback_idx)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _load_panel_recaps(final_root: Path) -> dict[str, str]:
    """
    Map crop_path -> recap text, if `final/panel_recaps.jsonl` exists (Stage 3 output).
    """
    recaps: dict[str, str] = {}
    jsonl = final_root / "panel_recaps.jsonl"
    if not jsonl.exists():
        return recaps
    for line in _read_text(jsonl).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        crop_path = obj.get("crop_path")
        recap = obj.get("recap")
        if isinstance(crop_path, str) and isinstance(recap, str) and crop_path.strip() and recap.strip():
            recaps[crop_path] = recap.strip()
    return recaps


def _chunk(items: list[dict], size: int) -> list[list[dict]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _speaker_lines_from_transcript(transcript_txt: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in (transcript_txt or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SPEAKER_TAG_RE.match(line)
        if not m:
            continue
        speaker = m.group(1).strip()
        text = m.group(2).strip()
        if not speaker or not text:
            continue
        out.append((speaker, text))
    return out


def _extract_candidate_names(all_text: str) -> set[str]:
    text = all_text or ""
    candidates: set[str] = set()

    for m in _TITLED_NAME_RE.finditer(text):
        title = m.group(1).strip()
        name = m.group(2).strip()
        if title and name:
            candidates.add(f"{title} {name}")

    for tok in re.findall(r"\b[A-Z][a-z]{2,}\b", text):
        candidates.add(tok.strip())

    stop = {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "Once",
        "Well",
        "When",
        "What",
        "Why",
        "How",
        "Stop",
        "Watch",
        "Someone",
        "Respect",
        "Class",
        "Grade",
        "Japanese",
        "Maam",
    }
    return {c for c in candidates if c and c not in stop}


def _guess_character_map(*, panel_dirs: list[Path]) -> dict[str, str]:
    all_transcripts: list[str] = []
    by_speaker: dict[str, list[str]] = {}

    for d in panel_dirs:
        t = _read_text(d / "transcript.txt").strip()
        if not t:
            continue
        all_transcripts.append(t)
        for speaker, text in _speaker_lines_from_transcript(t):
            by_speaker.setdefault(speaker, []).append(text)

    blob = "\n".join(all_transcripts)
    _candidates = _extract_candidate_names(blob)

    main_name = ""
    m = _BY_NAME_RE.search(blob)
    if m:
        main_name = m.group(1).strip()

    akane_like: dict[str, int] = {}
    father_like: dict[str, int] = {}
    for speaker, lines in by_speaker.items():
        s = "\n".join(lines)
        ak = 0
        fa = 0
        if main_name:
            if re.search(rf"\bby\s+{re.escape(main_name)}\b", s):
                ak += 10
            if re.search(r"\bStop it,\s*Dad\b", s, flags=re.IGNORECASE) or re.search(r"\bDad!!\b", s):
                ak += 6
            if re.search(rf"\bLet's go,\s*{re.escape(main_name)}\b", s):
                fa += 8
            if re.search(rf"\b{re.escape(main_name)}\b", s) and "by " not in s:
                fa += 1
        akane_like[speaker] = ak
        father_like[speaker] = fa

    out: dict[str, str] = {}
    if main_name and akane_like:
        ak_speaker = max(akane_like.items(), key=lambda kv: kv[1])[0]
        if akane_like.get(ak_speaker, 0) > 0:
            out[ak_speaker] = main_name

    if father_like:
        fa_speaker = max(father_like.items(), key=lambda kv: kv[1])[0]
        if father_like.get(fa_speaker, 0) > 0 and out.get(fa_speaker) != main_name:
            out[fa_speaker] = f"{main_name}'s father" if main_name else "the father"

    return out


def _format_character_map(character_map: dict[str, str]) -> str:
    if not character_map:
        return "(none)"
    lines = []
    for k in sorted(character_map.keys()):
        lines.append(f"- {k}: {character_map[k]}")
    return "\n".join(lines).strip()


def _build_raw_panel_block(
    *,
    panels: list[dict],
    out_root: Path,
    recaps_from_jsonl: dict[str, str],
    character_map: dict[str, str],
) -> str:
    blocks: list[str] = []
    for p in panels:
        crop_rel = p.get("crop_path")
        if not isinstance(crop_rel, str) or not crop_rel.strip():
            continue
        crop_path = out_root / crop_rel
        if not crop_path.exists():
            continue
        panel_dir = crop_path.parent

        scene_caption = p.get("scene_caption") if isinstance(p.get("scene_caption"), str) else ""
        transcript_txt = _read_text(panel_dir / "transcript.txt").strip()

        lines_out: list[str] = []
        if transcript_txt:
            for speaker, text in _speaker_lines_from_transcript(transcript_txt):
                sp = character_map.get(speaker, speaker)
                lines_out.append(f"{sp}: {text}")
        else:
            recap = recaps_from_jsonl.get(crop_rel, "").strip()
            if not recap:
                recap = _read_text(panel_dir / "recap.txt").strip()
            if recap:
                lines_out.append(recap)

        if not (scene_caption.strip() or lines_out):
            continue

        chunk = ""
        if scene_caption.strip():
            chunk += f"Scene: {scene_caption.strip()}\n"
        if lines_out:
            chunk += "\n".join(lines_out).strip()
        blocks.append(chunk.strip())

    return "\n\n".join(blocks).strip()


def _build_prompt(*, previous_chunk_recap: str, raw_panel_block: str, character_map: dict[str, str]) -> str:
    return (
        "Known characters (use these names consistently):\n"
        f"{_format_character_map(character_map)}\n\n"
        "Previous section (for continuity — do NOT repeat or summarize this, \n"
        "just continue from it):\n"
        f"{(previous_chunk_recap or '').strip()}\n\n"
        "Write a recap of these panels:\n"
        f"{(raw_panel_block or '').strip()}\n"
    )


GROQ_SYSTEM_PROMPT = """You are a sharp, concise manga recap writer. Your job is to retell 
what literally happens in a set of manga panels as clean, engaging 
prose — like a skilled narrator summarizing a story for a reader.

Strict rules:
- Never use meta-commentary like "the story unfolds", "the scene shifts", 
  "connections are becoming clearer", or "building to a climax"
- Never speculate about what things "might mean" or "possibly suggest"
- Never pad with filler — if you have nothing new to say, stop writing
- Write ONLY what concretely happens: who does what, who says what, 
  what changes
- Use past tense, third person
- Keep it tight — 3 to 5 sentences per chunk maximum
- Preserve character names, specific dialogue, and emotional tone exactly
- Output only the recap text, no labels, no preamble, nothing else
"""


def _groq_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout_s: int,
    max_tokens: int,
    temperature: float,
) -> str:
    import requests  # local import to keep 'none' mode dependency-free

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": GROQ_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    r = requests.post(url, headers=headers, json=payload, timeout=max(5, int(timeout_s)))
    r.raise_for_status()
    data = r.json()
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except Exception:
        return str(data).strip()


def main() -> None:
    _maybe_load_dotenv()
    ap = argparse.ArgumentParser(
        description=(
            "Stage 4: rewrite raw per-panel recaps into a coherent, flowing chapter recap in chunks (3–5 panels), "
            "optionally using Groq (OpenAI-compatible) for the rewrite."
        )
    )
    ap.add_argument("storyboard", help="Path to final/storyboard.json")

    ap.add_argument("--provider", choices=["none", "groq"], default="groq")
    ap.add_argument("--chunk-size", type=int, default=4, help="Panels per chunk (recommended: 3–5).")
    ap.add_argument("--context-chunks", type=int, default=1, help="How many previous chunk outputs to include as context.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing final/recap_final.txt if present.")
    ap.add_argument(
        "--character-map-json",
        default=None,
        help="Optional JSON file mapping speaker tags (e.g. char_0) to names (overrides auto-guess).",
    )

    ap.add_argument("--groq-key-env", default="GROQ_SCRIPT_KEY")
    ap.add_argument("--groq-model", default=os.environ.get("GROQ_SCRIPT_MODEL", ""))
    ap.add_argument("--groq-base-url", default=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    ap.add_argument("--groq-timeout", type=int, default=120)
    ap.add_argument("--groq-max-tokens", type=int, default=700)
    ap.add_argument("--groq-temperature", type=float, default=0.2)

    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--log-every", type=int, default=1)
    args = ap.parse_args()

    in_path = Path(args.storyboard)
    if not in_path.exists():
        _fail(f"File not found: {in_path}")
    final_root = in_path.parent
    out_root = final_root.parent

    out_final = final_root / "recap_final.txt"
    out_chunks_dir = final_root / "recap_chunks"
    out_chunks_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = final_root / "recap_chunks.jsonl"

    if out_final.exists() and not bool(args.overwrite):
        print(f"final recap already exists; pass --overwrite to replace it: {out_final}")
        return

    doc = json.loads(in_path.read_text(encoding="utf-8"))
    panels = doc.get("panels")
    if not isinstance(panels, list) or not panels:
        _fail("storyboard.panels must be a non-empty array.")

    recaps_from_jsonl = _load_panel_recaps(final_root)

    panels_sorted: list[dict] = []
    for idx, p in enumerate(panels):
        if isinstance(p, dict):
            pp = dict(p)
            pp["_order_idx"] = idx
            panels_sorted.append(pp)
    panels_sorted = sorted(panels_sorted, key=lambda p: _panel_sort_key(p, int(p.get("_order_idx") or 0)))

    panel_dirs: list[Path] = []
    for p in panels_sorted:
        crop_rel = p.get("crop_path")
        if not isinstance(crop_rel, str) or not crop_rel.strip():
            continue
        crop_path = out_root / crop_rel
        if not crop_path.exists():
            continue
        panel_dirs.append(crop_path.parent)

    character_map = _guess_character_map(panel_dirs=panel_dirs)
    if args.character_map_json:
        try:
            override = json.loads(Path(args.character_map_json).read_text(encoding="utf-8"))
        except Exception:
            override = None
        if isinstance(override, dict):
            for k, v in override.items():
                if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                    character_map[k.strip()] = v.strip()
    try:
        (final_root / "character_map.json").write_text(json.dumps(character_map, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    chunk_size = int(args.chunk_size)
    if not (3 <= chunk_size <= 5):
        print(f"Warning: --chunk-size is {chunk_size}; recommended is 3–5.")

    chunks = _chunk(panels_sorted, chunk_size)

    api_key = ""
    if args.provider == "groq":
        api_key = os.environ.get(args.groq_key_env, "").strip()
        # Convenience fallbacks for common env var names.
        if not api_key and str(args.groq_key_env) == "GROQ_SCRIPT_KEY":
            api_key = os.environ.get("GROQ_API_KEY", "").strip() or os.environ.get("GROQ_KEY", "").strip()
        if not api_key:
            _fail(
                f"Missing Groq API key env var {args.groq_key_env!r}. "
                "If your key is stored in GROQ_API_KEY, re-run with: --groq-key-env GROQ_API_KEY"
            )
        if not args.groq_model:
            # Try common fallbacks as well.
            env_model = os.environ.get("GROQ_SCRIPT_MODEL", "").strip() or os.environ.get("GROQ_MODEL", "").strip()
            if env_model:
                args.groq_model = env_model
            else:
                _fail("Missing Groq model. Pass --groq-model or set GROQ_SCRIPT_MODEL.")

    if args.progress:
        print(f"[FinalRecap] provider={args.provider} chunks={len(chunks)} chunk_size={chunk_size}")
        sys.stdout.flush()

    chunk_outputs: list[dict] = []
    start_all = time.time()
    for idx, ch in enumerate(chunks):
        story_so_far = "\n\n".join(
            [
                x.get("text", "")
                for x in chunk_outputs[-max(0, int(args.context_chunks)) :]
                if isinstance(x, dict) and isinstance(x.get("text"), str) and x.get("text")
            ]
        ).strip()
        raw_panel_block = _build_raw_panel_block(
            panels=ch,
            out_root=out_root,
            recaps_from_jsonl=recaps_from_jsonl,
            character_map=character_map,
        )
        if not raw_panel_block:
            continue
        prompt = _build_prompt(previous_chunk_recap=story_so_far, raw_panel_block=raw_panel_block, character_map=character_map)

        if args.progress and (idx == 0 or (idx + 1) % max(1, int(args.log_every)) == 0 or idx + 1 == len(chunks)):
            print(f"[FinalRecap] ({idx+1}/{len(chunks)}) rewriting {len(ch)} panels")
            sys.stdout.flush()

        t0 = time.time()
        if args.provider == "none":
            text = " ".join(raw_panel_block.split()).strip()
        else:
            text = _groq_chat(
                base_url=str(args.groq_base_url),
                api_key=api_key,
                model=str(args.groq_model),
                prompt=prompt,
                timeout_s=int(args.groq_timeout),
                max_tokens=int(args.groq_max_tokens),
                temperature=float(args.groq_temperature),
            )
        dt = time.time() - t0

        chunk_id = f"c{idx:04d}"
        chunk_obj = {
            "chunk_id": chunk_id,
            "panel_ids": [x.get("panel_id", "") for x in ch if isinstance(x, dict)],
            "crop_paths": [x.get("crop_path", "") for x in ch if isinstance(x, dict)],
            "text": (text or "").strip(),
            "provider": args.provider,
            "model": args.groq_model if args.provider == "groq" else "",
            "generated_at": _now_utc_iso(),
            "elapsed_s": round(dt, 3),
        }
        chunk_outputs.append(chunk_obj)

        (out_chunks_dir / f"{chunk_id}.txt").write_text(chunk_obj["text"] + ("\n" if chunk_obj["text"] else ""), encoding="utf-8")
        (out_chunks_dir / f"{chunk_id}.raw.json").write_text(
            json.dumps(
                {"prompt": prompt, "raw_panel_block": raw_panel_block, "character_map": character_map, "panels": ch},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    out_jsonl.write_text("\n".join([json.dumps(x, ensure_ascii=False) for x in chunk_outputs]) + "\n", encoding="utf-8")
    final_text = "\n\n".join([x.get("text", "").strip() for x in chunk_outputs if x.get("text")]).strip()
    out_final.write_text(final_text + ("\n" if final_text else ""), encoding="utf-8")

    if args.progress:
        dt_all = time.time() - start_all
        print(f"[FinalRecap] wrote: {out_final}")
        print(f"[FinalRecap] wrote: {out_jsonl}")
        print(f"[FinalRecap] wrote: {out_chunks_dir}")
        print(f"[FinalRecap] done in {dt_all/60.0:.1f} min")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
