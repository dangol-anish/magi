#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_SPEAKER_LINE_RE = re.compile(r"^<([^>]+)>:\s*(.*)\s*$")
_WORD_RE = re.compile(r"[A-Za-z']+")

# Common self-ID patterns.
_RE_MY_NAME_IS = re.compile(r"\bmy name is\s+([A-Z][a-z]+)\b")
_RE_I_AM = re.compile(r"\bi am\s+([A-Z][a-z]+)\b")
_RE_IM = re.compile(r"\bi'?m\s+([A-Z][a-z]+)\b")
_RE_CALL_ME = re.compile(r"\bcall me\s+([A-Z][a-z]+)\b")

# Address patterns: "Akane," or "Akane!" or "Hey Akane"
_RE_NAME_VOCATIVE = re.compile(r"\b([A-Z][a-z]{2,})[,!]\b")
_RE_HEY_NAME = re.compile(r"\bhey\s+([A-Z][a-z]{2,})\b", flags=re.IGNORECASE)

# Titles: "Mr. Yanagiya"
_RE_TITLED = re.compile(r"\b(Mr\.|Ms\.|Mrs\.|Dr\.)\s+([A-Z][a-z]{2,})\b")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _parse_transcript_txt(text: str) -> list[dict]:
    items: list[dict] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SPEAKER_LINE_RE.match(line)
        if not m:
            continue
        speaker = m.group(1).strip()
        utter = m.group(2).strip()
        if not speaker or not utter:
            continue
        items.append({"speaker": speaker, "text": utter})
    return items


def _load_panel_transcript(panel_dir: Path) -> list[dict]:
    """
    Returns list of {speaker, text}.
    """
    try:
        data = json.loads(_read_text(panel_dir / "transcript.json"))
    except Exception:
        data = None

    if isinstance(data, list):
        out: list[dict] = []
        for it in data:
            if not isinstance(it, dict):
                continue
            speaker = it.get("speaker")
            text = it.get("text")
            if isinstance(speaker, str) and isinstance(text, str) and speaker.strip() and text.strip():
                out.append({"speaker": speaker.strip(), "text": text.strip()})
        if out:
            return out

    return _parse_transcript_txt(_read_text(panel_dir / "transcript.txt"))


def _candidate_names_from_text(text: str) -> set[str]:
    """
    Very conservative candidate extraction: ProperCase tokens and titled names.
    """
    t = text or ""
    out: set[str] = set()
    for m in _RE_TITLED.finditer(t):
        out.add(f"{m.group(1)} {m.group(2)}")
        out.add(m.group(2))
    for w in re.findall(r"\b[A-Z][a-z]{2,}\b", t):
        out.add(w)

    # Remove common sentence starters / non-names.
    stop = {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "Once",
        "When",
        "What",
        "Why",
        "How",
        "Someone",
        "Respect",
        "Japanese",
        "Class",
        "Grade",
        "Watch",
        "Stop",
        "Dad",
        "Mom",
        "Maam",
        "Okay",
    }
    return {x for x in out if x and x not in stop}


def _score_speaker_names(*, speaker_lines: list[str]) -> tuple[str, dict]:
    """
    Returns (best_name, evidence_dict).
    """
    # Collect explicit self-identification
    explicit: Counter[str] = Counter()
    for line in speaker_lines:
        for rx in (_RE_MY_NAME_IS, _RE_I_AM, _RE_IM, _RE_CALL_ME):
            m = rx.search(line)
            if m:
                explicit[m.group(1)] += 1

    if explicit:
        name, cnt = explicit.most_common(1)[0]
        return name, {"reason": "self_identification", "hits": dict(explicit)}

    # Else: if a speaker frequently says "Stop it, Dad" or similar, that's likely the child.
    dad_hits = sum(1 for ln in speaker_lines if re.search(r"\bDad\b", ln))
    mom_hits = sum(1 for ln in speaker_lines if re.search(r"\bMom\b", ln))
    if dad_hits or mom_hits:
        role = "child"
        return role, {"reason": "kinship_terms", "dad_hits": dad_hits, "mom_hits": mom_hits}

    return "", {"reason": "no_strong_signal"}


def _extract_addressed_names(*, all_lines: list[str]) -> Counter[str]:
    """
    Names that appear as direct address in dialogue.
    """
    c: Counter[str] = Counter()
    for line in all_lines:
        for m in _RE_NAME_VOCATIVE.finditer(line):
            c[m.group(1)] += 1
        m2 = _RE_HEY_NAME.search(line)
        if m2:
            c[m2.group(1)] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Infer a consistent speaker->character name map by scanning all panel transcripts in a chapter. "
            "This is best-effort and intended to reduce 'char_0/char_1' confusion in later recap stages."
        )
    )
    ap.add_argument("storyboard", help="Path to final/storyboard.json")
    ap.add_argument(
        "--out",
        default=None,
        help="Write mapping JSON to this path (default: final/character_map.json).",
    )
    ap.add_argument(
        "--report",
        default=None,
        help="Write a debug report JSON to this path (default: final/character_map_report.json).",
    )
    args = ap.parse_args()

    in_path = Path(args.storyboard)
    if not in_path.exists():
        _fail(f"File not found: {in_path}")

    final_root = in_path.parent
    out_root = final_root.parent

    doc = json.loads(in_path.read_text(encoding="utf-8"))
    panels = doc.get("panels")
    if not isinstance(panels, list) or not panels:
        _fail("storyboard.panels must be a non-empty array.")

    # Gather all transcript lines across all panels.
    by_speaker: dict[str, list[str]] = defaultdict(list)
    all_lines: list[str] = []
    panel_dirs_seen: set[str] = set()

    for p in panels:
        if not isinstance(p, dict):
            continue
        crop_rel = p.get("crop_path")
        if not isinstance(crop_rel, str) or not crop_rel.strip():
            continue
        crop_path = out_root / crop_rel
        panel_dir = crop_path.parent
        if str(panel_dir) in panel_dirs_seen:
            continue
        panel_dirs_seen.add(str(panel_dir))

        items = _load_panel_transcript(panel_dir)
        for it in items:
            speaker = it.get("speaker")
            text = it.get("text")
            if not (isinstance(speaker, str) and isinstance(text, str)):
                continue
            speaker = speaker.strip()
            text = text.strip()
            if not speaker or not text:
                continue
            by_speaker[speaker].append(text)
            all_lines.append(text)

    if not by_speaker:
        _fail("No transcript lines found. Run Stage 1 first (it writes transcript.txt/json per panel).")

    addressed = _extract_addressed_names(all_lines=all_lines)
    candidates = _candidate_names_from_text("\n".join(all_lines))

    # Per-speaker name inference.
    mapping: dict[str, str] = {}
    report: dict = {
        "generated_at": _now_utc_iso(),
        "speakers": {},
        "addressed_names": addressed.most_common(30),
        "candidates": sorted(candidates)[:200],
    }

    # First pass: explicit self-ID or role.
    provisional_roles: dict[str, str] = {}
    for speaker, lines in by_speaker.items():
        best, evidence = _score_speaker_names(speaker_lines=lines)
        if best:
            provisional_roles[speaker] = best
        report["speakers"][speaker] = {
            "line_count": len(lines),
            "top_words": Counter([w.lower() for ln in lines for w in _WORD_RE.findall(ln)]).most_common(12),
            "evidence": evidence,
        }

    # Second pass: pick a main character name (most-addressed proper name).
    main_name = ""
    for name, _cnt in addressed.most_common(20):
        if name in candidates:
            main_name = name
            break

    # If we found a main_name and there is a speaker labeled "child", map it to main_name.
    if main_name:
        for speaker, role in provisional_roles.items():
            if role == "child" and speaker not in mapping:
                mapping[speaker] = main_name

    # If any speaker has a self-identified name, map it directly.
    for speaker, lines in by_speaker.items():
        best, ev = _score_speaker_names(speaker_lines=lines)
        if ev.get("reason") == "self_identification" and isinstance(best, str) and best:
            mapping[speaker] = best

    # If we have a main_name but no mapped speaker, try to use the strongest OCR cue:
    # choose the speaker that mentions "by <main_name>" or mentions main_name often.
    if main_name and all(v != main_name for v in mapping.values()):
        scores: dict[str, int] = {}
        for speaker, lines in by_speaker.items():
            s = "\n".join(lines)
            score = 0
            if re.search(rf"\bby\s+{re.escape(main_name)}\b", s):
                score += 10
            score += len(re.findall(rf"\b{re.escape(main_name)}\b", s))
            scores[speaker] = score
        best_speaker = max(scores.items(), key=lambda kv: kv[1])[0]
        if scores.get(best_speaker, 0) > 0:
            mapping[best_speaker] = main_name

    # Very light heuristic: if a speaker says "Let's go, <main_name>", they are likely "<main_name>'s father".
    if main_name:
        for speaker, lines in by_speaker.items():
            s = "\n".join(lines)
            if re.search(rf"\bLet's go,\s*{re.escape(main_name)}\b", s):
                if speaker not in mapping:
                    mapping[speaker] = f"{main_name}'s father"

    # Fill remaining speakers as-is (keeps later stages deterministic).
    for speaker in sorted(by_speaker.keys()):
        mapping.setdefault(speaker, speaker)

    out_path = Path(args.out) if args.out else (final_root / "character_map.json")
    report_path = Path(args.report) if args.report else (final_root / "character_map_report.json")
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"Wrote: {report_path}")
    print(f"- speakers: {len(mapping)}")
    if main_name:
        print(f"- main_name_guess: {main_name}")


if __name__ == "__main__":
    main()

