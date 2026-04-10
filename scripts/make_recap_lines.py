#!/usr/bin/env python3
import argparse
import json
import re
import os
import sys
from pathlib import Path


def _fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")

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


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])


def _looks_like_quote(text: str) -> bool:
    # Heuristic: disallow obvious quoting punctuation.
    t = text or ""
    return any(ch in t for ch in ['"', "“", "”", "’", "‘"])


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _trim_to_words(text: str, max_words: int) -> str:
    """
    Trim to <= max_words and avoid dangling trailing stop-words like "at/of/to".
    """
    t = " ".join((text or "").strip().split())
    if not t:
        return ""
    words = t.split()
    if len(words) <= max_words:
        return t

    words = words[:max_words]
    trailing_stop = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "but",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "to",
        "with",
        "while",
    }
    while words and words[-1].lower().strip(".,;:!?") in trailing_stop:
        words = words[:-1]

    # If we end on a connector phrase fragment like "but daily", drop it.
    if len(words) >= 2:
        last = words[-1].lower().strip(".,;:!?")
        prev = words[-2].lower().strip(".,;:!?")
        connectors = {"but", "so", "then", "yet", "still", "meanwhile", "even", "also", "and", "or"}
        time_words = {"daily", "today", "now", "lately", "tonight", "tomorrow", "yesterday", "always"}
        if prev in connectors and last in time_words:
            words = words[:-2]
        elif last in connectors:
            words = words[:-1]

    out = " ".join(words).rstrip(",;:-")
    # Drop dangling leading quote apostrophes in the last token (e.g. "'To")
    out = re.sub(r"\s+'\w+$", "", out).strip()
    if not out.endswith((".", "!", "?")):
        out += "."
    return out


def _offline_recap_for_beat(*, beat_panel_ids: list[str], scene_caps: list[str]) -> list[dict]:
    """
    Deterministic placeholder: 2 short narrator sentences, no quotes.
    Used for testing the pipeline without an LLM call.
    """
    cap = ""
    for c in scene_caps:
        c = (c or "").strip()
        if c:
            cap = c
            break

    if cap:
        s1 = cap
    else:
        s1 = "The scene continues, but details are unclear."
    s1 = " ".join(s1.split())
    if _looks_like_quote(s1):
        s1 = s1.replace('"', "").replace("“", "").replace("”", "")

    # Keep short.
    s1 = _trim_to_words(s1, 12)

    s2 = "Tension hangs in the air as the moment unfolds."
    # Distribute panels across sentences for fast cuts.
    p1 = beat_panel_ids[:1]
    p2 = beat_panel_ids[1:2] or p1
    return [{"text": s1, "panel_ids": p1}, {"text": s2, "panel_ids": p2}]


def _text_to_items(text: str, *, beat_panel_ids: list[str], max_items: int) -> list[dict]:
    """
    Best-effort converter when a provider doesn't return JSON.
    Splits into short sentences and assigns panels in reading order.
    """
    t = " ".join((text or "").strip().split())
    if not t:
        return []
    t = t.replace('"', "").replace("“", "").replace("”", "").replace("’", "'")
    # Split on sentence boundaries / newlines.
    raw = re.split(r"(?<=[.!?])\s+|\n+", t)
    sents = [s.strip() for s in raw if s and s.strip()]
    sents = sents[: max(1, int(max_items))]
    out: list[dict] = []
    for i, s in enumerate(sents):
        pid = beat_panel_ids[min(i, len(beat_panel_ids) - 1) : min(i, len(beat_panel_ids) - 1) + 1]
        out.append({"text": s, "panel_ids": pid})
    return out


def _contains_quote_like_apostrophe(text: str) -> bool:
    """
    Detect apostrophes that look like quoting rather than contractions.
    We allow contractions like "isn't" (apostrophe between letters),
    but reject standalone / leading apostrophes like "'But" or " 'But".
    """
    t = text or ""
    if '"' in t or "“" in t or "”" in t or "‘" in t or "’" in t:
        return True
    # Any apostrophe not between letters is considered quote-like.
    return re.search(r"(?<![A-Za-z])'(?![A-Za-z])|(^')|(\s')|('\s)", t) is not None


def _sanitize_sentence(text: str, *, max_words: int) -> str:
    """
    Enforce "no quotes" and word limit, and avoid cut-off endings.
    """
    t = " ".join((text or "").strip().split())
    if not t:
        return ""
    # Avoid awkward one-word comma asides like "A woman, beaming, ..." or "The man, mid-bite, ...".
    t = re.sub(r"^(\bA\b|\bAn\b|\bThe\b)\s+([^,]{1,60}),\s*([A-Za-z-]{2,20}),\s+", r"\1 \2 ", t)
    # Remove common quote characters.
    t = t.replace('"', "").replace("“", "").replace("”", "").replace("‘", "").replace("’", "'")
    # Remove apostrophes that are not between letters (likely quote artifacts).
    t = re.sub(r"(?<![A-Za-z])'(?![A-Za-z])", "", t)
    # Also remove any leading apostrophe (e.g. "'But").
    t = re.sub(r"^'+", "", t).strip()
    # Apply word limit.
    if _word_count(t) > max_words:
        t = _trim_to_words(t, max_words)
    return t


def _ocr_to_hints(ocr_lines: list[str]) -> list[str]:
    """
    Convert raw OCR lines into paraphrased, non-quoting "dialogue topic" hints.
    This reduces the chance the model copies dialogue verbatim.
    """
    hints: list[str] = []
    seen: set[str] = set()
    blob = " ".join([(x or "").strip() for x in ocr_lines if isinstance(x, str)]).lower()

    def _add(h: str) -> None:
        h = " ".join(h.split()).strip()
        if not h or h in seen:
            return
        hints.append(h)
        seen.add(h)

    if any(k in blob for k in ["husband", "understanding", "great husband"]):
        _add("She praises her husband's patience and understanding.")
    if any(k in blob for k in ["cook", "cooking", "great cook", "home-cooked", "home cooked"]):
        _add("She admits she struggles with cooking and feels insecure.")
    if any(k in blob for k in ["married", "marry"]):
        _add("She reflects on how small habits shape a relationship.")
    if any(k in blob for k in ["workload", "office", "work"]):
        _add("Work pressure weighs on him, and it slips into the conversation.")
    if any(k in blob for k in ["easier", "tastier"]):
        _add("A casual food comment turns into something more personal.")
    if any(k in blob for k in ["freeze"]):
        _add("He suddenly tenses up, sensing the mood change.")

    # If we couldn't detect anything, keep a single vague hint.
    if not hints and blob:
        _add("Their conversation carries an undercurrent of worry and relief.")

    return hints[:6]


def _looks_like_sfx(text: str) -> bool:
    """
    Heuristic filter for manga sound effects (SFX) like "Beam", "Freeze", "Chew Chew".
    These are often short, standalone tokens with no punctuation.
    """
    t = " ".join((text or "").strip().split())
    if not t:
        return False

    # Remove surrounding punctuation
    core = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    if not core:
        return False

    # Reject if it contains digits or many symbols; that's unlikely to be SFX-only
    if re.search(r"\d", core):
        return False

    words = core.split()
    if len(words) > 3:
        return False

    # If it has sentence punctuation, it's probably dialogue/narration, not SFX
    if any(ch in t for ch in [".", "!", "?", ":", ";", ","]):
        return False

    # Typical SFX are title-case or all-caps (but not always).
    is_title_or_caps = all(w.isalpha() and (w.istitle() or w.isupper()) for w in words)
    if not is_title_or_caps:
        return False

    # Ignore common non-SFX tokens
    ignore = {"I", "A", "An", "The", "And", "But", "So", "Then"}
    if core in ignore:
        return False

    # Examples: Beam, Freeze, Chew Chew, Slam, Crack
    return True


def _filter_ocr_lines(ocr_lines: list[str]) -> list[str]:
    out: list[str] = []
    for x in ocr_lines:
        if not isinstance(x, str):
            continue
        t = " ".join(x.strip().split())
        if not t:
            continue
        if _looks_like_sfx(t):
            continue
        out.append(t)
    return out


def _clean_ocr_story_lines(ocr_lines: list[str], *, min_words: int = 4) -> list[str]:
    """
    Extra OCR cleanup for story evidence: drop pure SFX and short fragments.
    Keep short lines if they look like complete sentences (end punctuation).
    """
    out: list[str] = []
    for x in ocr_lines:
        if not isinstance(x, str):
            continue
        t = " ".join(x.strip().split())
        if not t:
            continue
        if _looks_like_sfx(t):
            continue
        if _word_count(t) < min_words and not re.search(r"[.!?]$", t):
            continue
        out.append(t)
    return out


def _normalize_problematic_names(texts: list[str], name_map: dict[str, str]) -> list[str]:
    """
    Replace OCR-misread "names" that collide with common English words (e.g. "You").
    """
    if not name_map:
        return texts
    out: list[str] = []
    for t in texts:
        s = t
        for bad, good in name_map.items():
            if not bad or not good:
                continue
            s = re.sub(rf"\b{re.escape(bad)}\b", good, s)
        out.append(s)
    return out


def _ocr_to_story_facts(ocr_lines: list[str]) -> list[str]:
    """
    Convert raw OCR lines into story-meaning facts (paraphrased, no quotes).
    This is the highest-priority evidence for recap writing.
    """
    raw = [x for x in ocr_lines if isinstance(x, str)]
    blob = " ".join(raw).lower()

    # Extract potential names (simple heuristic).
    name_candidates: list[str] = []
    for line in raw:
        for m in re.findall(r"\b[A-Z][a-z]{2,}\b", line):
            if m.lower() in {"the", "and", "but", "then", "meanwhile", "still", "that"}:
                continue
            name_candidates.append(m)
    # Dedupe while preserving order
    names: list[str] = []
    seen_names: set[str] = set()
    for n in name_candidates:
        if n not in seen_names:
            names.append(n)
            seen_names.add(n)

    facts: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = " ".join(s.split()).strip()
        if not s or s in seen:
            return
        facts.append(s)
        seen.add(s)

    # General name grounding.
    if names:
        if len(names) == 1:
            _add(f"Key name mentioned: {names[0]}.")
        else:
            _add(f"Key names mentioned: {', '.join(names[:3])}.")

    # Relationship / admiration.
    if "husband" in blob or "wife" in blob:
        _add("A couple talks about their relationship and what makes it work.")
    if any(k in blob for k in ["understanding", "great husband"]):
        if names:
            _add(f"She praises {names[0]} for being patient and understanding.")
        else:
            _add("She praises her partner for being patient and understanding.")

    # Cooking admission + implied stakes.
    if any(k in blob for k in ["not a great cook", "not great cook", "i'm not a great cook", "im not a great cook"]):
        _add("She admits she is not very good at cooking.")
    if any(k in blob for k in ["home-cooked", "home cooked", "insisted on home", "every day"]):
        _add("She thinks daily home-cooked demands would have been a dealbreaker.")
    if any(k in blob for k in ["might not have married", "wouldn't have married", "would not have married"]):
        _add("She hints she might not have married him under different expectations.")

    # Work stress / tonal shift.
    if any(k in blob for k in ["workload", "office", "at the office", "been increasing"]):
        _add("Work stress starts creeping into the dinner conversation.")

    # Hard cue / tension.
    if "freeze" in blob:
        _add("Someone suddenly goes still, and the mood shifts.")

    # If we still have nothing, keep one generic but useful fact.
    if not facts:
        _add("The evidence is unclear, but the scene suggests a change in mood.")

    return facts[:8]

def _simplify_for_kid(text: str) -> str:
    """
    Best-effort simplifier. Keeps names as-is by only replacing common words.
    """
    t = " ".join((text or "").strip().split())
    if not t:
        return ""

    replacements = {
        "radiant": "happy",
        "beaming": "smiling",
        "vulnerability": "sadness",
        "confession": "admission",
        "insecurities": "worries",
        "ponder": "think",
        "ponders": "thinks",
        "reflect": "think",
        "reflects": "thinks",
        "pressure": "stress",
        "pressures": "stress",
        "overwhelmed": "very tired",
        "contemplative": "quiet",
        "teasing": "hinting",
        "exploration": "talk",
        "unexpected": "surprising",
        "conversation": "talk",
        "reveals": "shows",
        "admits": "says",
        "appreciates": "likes",
        "acceptance": "kindness",
        "tenses": "gets stiff",
        "tensing": "getting stiff",
        "habit": "thing",
        "habits": "things",
        "strengthen": "make stronger",
        "bond": "friendship",
        "marital": "family",
    }

    # Replace whole words/phrases case-insensitively, preserving capitalization.
    for src, dst in replacements.items():
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)

        def _repl(m):
            w = m.group(0)
            rep = dst
            if w.isupper():
                return rep.upper()
            if w[:1].isupper():
                return rep[:1].upper() + rep[1:]
            return rep

        t = pattern.sub(_repl, t)

    return " ".join(t.split())


def _simplify_for_simple(text: str) -> str:
    """
    Simpler than normal, but not baby talk.
    """
    t = " ".join((text or "").strip().split())
    if not t:
        return ""

    replacements = {
        "meanwhile": "at the same time",
        "attentively": "carefully",
        "acknowledging": "showing",
        "reassures": "calms",
        "reassure": "calm",
        "expression": "face",
        "softening": "relaxing",
        "hardening": "going serious",
        "weighs on him": "bothers him",
        "weighs on her": "bothers her",
        "work pressure": "work stress",
        "pressure": "stress",
        "echoing": "filling the air",
        "distant": "far away",
        "bond": "relationship",
        "impact": "effect",
        "habit": "small habit",
        "habits": "small habits",
        "glances": "looks",
        "gaze": "look",
    }

    for src, dst in replacements.items():
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)

        def _repl(m):
            w = m.group(0)
            rep = dst
            if w.isupper():
                return rep.upper()
            if w[:1].isupper():
                return rep[:1].upper() + rep[1:]
            return rep

        t = pattern.sub(_repl, t)

    return " ".join(t.split())

def _split_to_min_sentences(
    items: list[dict], *, min_items: int, beat_panel_ids: list[str], max_words: int
) -> list[dict]:
    """
    If a provider returns too few sentences, split the longest sentence into multiple
    short ones instead of falling back to generic boilerplate.
    """
    if len(items) >= min_items:
        return items
    if not items:
        return items

    longest_idx = max(range(len(items)), key=lambda i: _word_count(str(items[i].get("text", ""))))
    text = " ".join(str(items[longest_idx].get("text", "")).split())
    words = [w for w in text.split() if w]
    if len(words) <= 1:
        return items

    chunks: list[str] = []
    for i in range(0, len(words), max(1, int(max_words))):
        chunk = " ".join(words[i : i + max_words]).strip()
        if chunk:
            if not chunk.endswith((".", "!", "?")):
                chunk += "."
            chunks.append(chunk)

    if not chunks:
        return items

    base = items[:longest_idx] + items[longest_idx + 1 :]
    out: list[dict] = []
    out.append({"text": chunks[0], "panel_ids": items[longest_idx].get("panel_ids") or beat_panel_ids[:1]})
    for c in chunks[1:]:
        if len(base) + len(out) >= min_items:
            break
        pidx = min(len(out), max(0, len(beat_panel_ids) - 1))
        out.append({"text": c, "panel_ids": beat_panel_ids[pidx : pidx + 1]})

    return base + out


def _build_evidence(panel: dict, *, max_ocr_lines: int) -> tuple[list[str], list[str], list[str]]:
    scene_caps: list[str] = []
    scene_tags: list[str] = []
    ocr_lines: list[str] = []

    cap = panel.get("scene_caption")
    if isinstance(cap, str) and cap.strip():
        scene_caps.append(cap.strip())

    tags = panel.get("scene_tags")
    if isinstance(tags, list):
        for t in tags:
            if isinstance(t, str) and t.strip():
                scene_tags.append(t.strip().lower())

    ocr = panel.get("ocr_lines")
    if isinstance(ocr, list):
        for ln in ocr:
            if not isinstance(ln, dict):
                continue
            t = ln.get("text")
            if isinstance(t, str) and t.strip():
                ocr_lines.append(" ".join(t.strip().split()))

    if max_ocr_lines > 0:
        ocr_lines = ocr_lines[:max_ocr_lines]

    return scene_caps, scene_tags, ocr_lines


def _bridge_sentence(*, scene_tags: list[str], ocr_lines: list[str], scene_caps: list[str], max_words: int) -> str:
    """
    Produce a short continuity sentence without quoting dialogue.
    Uses simple heuristics from evidence to avoid generic boilerplate.
    """
    tags = {t.lower() for t in scene_tags if isinstance(t, str)}
    text_blob = " ".join([*ocr_lines, *scene_caps]).lower()

    if any(x in text_blob for x in ["workload", "office", "work", "job"]):
        return _trim_to_words("The conversation shifts toward work pressures and quiet worries.", max_words)
    if any(x in text_blob for x in ["cook", "cooking", "meal", "home-cooked"]):
        return _trim_to_words("She wrestles with insecurity, but his patience steadies her.", max_words)
    if "freeze" in text_blob or "tense" in text_blob:
        return _trim_to_words("He tenses up, sensing the mood changing fast.", max_words)
    if "happy" in tags or "smile" in text_blob or "beam" in text_blob:
        return _trim_to_words("Her smile says more than the room ever could.", max_words)
    if "talking" in tags or "conversation" in tags:
        return _trim_to_words("Small talk turns meaningful in the middle of the meal.", max_words)

    return _trim_to_words("The tension builds as the scene quietly shifts.", max_words)


def _extract_json_blob(text: str) -> str:
    """
    Strip markdown fences and try to isolate a JSON object/array substring.
    """
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"```(?:json)?", "", t, flags=re.IGNORECASE).strip()
    # Try object first, then array.
    first_obj = None
    first_arr = None
    try:
        # naive scan for first {...}
        start = t.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(t)):
                ch = t[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        first_obj = t[start : i + 1]
                        break
    except Exception:
        first_obj = None
    try:
        start = t.find("[")
        if start != -1:
            depth = 0
            for i in range(start, len(t)):
                ch = t[i]
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        first_arr = t[start : i + 1]
                        break
    except Exception:
        first_arr = None

    return first_obj or first_arr or t


def _assign_panels_to_sentence(
    *, beat_panel_ids: list[str], sentence_idx: int, max_panels_per_sentence: int
) -> list[str]:
    """
    Distribute panel_ids across sentences in reading order for quick cuts.
    """
    max_panels_per_sentence = max(1, int(max_panels_per_sentence))
    if not beat_panel_ids:
        return []
    start = sentence_idx * max_panels_per_sentence
    end = start + max_panels_per_sentence
    if start >= len(beat_panel_ids):
        # repeat last panel if we run out
        return beat_panel_ids[-1:]
    return beat_panel_ids[start:end]


def _parse_beat_output(raw_text: str, *, beat_panel_ids: list[str]) -> dict:
    """
    Parse provider output into a dict:
      {"paragraph": str, "panel_ids": list[str]}
    Accepts either:
      - new format: JSON object {"text": "...", "panel_ids": [...]}
      - legacy format: JSON array [{"text": "...", "panel_ids":[...]}]
      - plain text: treated as paragraph, panel_ids defaults to beat_panel_ids
    """
    blob = _extract_json_blob(raw_text)
    if not blob:
        return {"paragraph": "", "panel_ids": beat_panel_ids}

    try:
        parsed = json.loads(blob)
    except Exception:
        # last resort: treat as plain paragraph
        return {"paragraph": " ".join((raw_text or "").split()).strip(), "panel_ids": beat_panel_ids}

    if isinstance(parsed, dict):
        text = parsed.get("text")
        panel_ids = parsed.get("panel_ids")
        paragraph = " ".join(str(text or "").split()).strip()
        if not paragraph:
            return {"paragraph": "", "panel_ids": beat_panel_ids}
        if not isinstance(panel_ids, list):
            panel_ids = beat_panel_ids
        panel_ids = [x for x in panel_ids if isinstance(x, str) and x in beat_panel_ids]
        if not panel_ids:
            panel_ids = beat_panel_ids
        return {"paragraph": paragraph, "panel_ids": panel_ids}

    if isinstance(parsed, list):
        # Legacy: join into a single paragraph so we regain flow.
        parts: list[str] = []
        panel_ids_out: list[str] = []
        for it in parsed:
            if not isinstance(it, dict):
                continue
            txt = it.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(" ".join(txt.strip().split()))
            pids = it.get("panel_ids")
            if isinstance(pids, list):
                for pid in pids:
                    if isinstance(pid, str) and pid in beat_panel_ids and pid not in panel_ids_out:
                        panel_ids_out.append(pid)
        paragraph = " ".join(parts).strip()
        return {"paragraph": paragraph, "panel_ids": panel_ids_out or beat_panel_ids}

    return {"paragraph": "", "panel_ids": beat_panel_ids}

def main() -> None:
    _maybe_load_dotenv()
    ap = argparse.ArgumentParser(
        description="Stage 2 (recap): generate YouTube-style narrator script lines per beat, linked to panel_ids."
    )
    ap.add_argument("storyboard", help="Path to final/storyboard.json")
    ap.add_argument("--out", default=None, help="Write to a different path (default: overwrite input).")
    ap.add_argument("--overwrite", action="store_true", help="Replace existing script[].")

    ap.add_argument("--provider", choices=["none", "groq", "gemini", "ollama", "auto"], default="none")
    ap.add_argument("--sentences-min", type=int, default=2)
    ap.add_argument("--sentences-max", type=int, default=4)
    ap.add_argument("--max-words", type=int, default=12)
    ap.add_argument("--max-panels-per-sentence", type=int, default=2)
    ap.add_argument("--max-ocr-lines", type=int, default=6, help="Max OCR lines to include as evidence per panel.")
    ap.add_argument(
        "--debug-raw",
        action="store_true",
        help="Embed provider raw output (truncated) into beats[].notes for debugging.",
    )
    ap.add_argument(
        "--reading-level",
        choices=["normal", "simple", "kid"],
        default="normal",
        help="Language complexity for narration. 'kid' aims for simple words.",
    )

    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    ap.add_argument("--gemini-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--gemini-log-key", action="store_true")
    ap.add_argument("--gemini-timeout", type=int, default=120)
    ap.add_argument("--groq-key-env", default="GROQ_SCRIPT_KEY")
    ap.add_argument("--groq-model", default=os.environ.get("GROQ_SCRIPT_MODEL", ""))
    ap.add_argument("--groq-base-url", default=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    ap.add_argument("--groq-timeout", type=int, default=120)
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument(
        "--ollama-model",
        default="mistral:7b-instruct",
        help="Ollama model to use for recap narration (text-instruct models recommended).",
    )
    ap.add_argument("--ollama-timeout", type=int, default=120)
    args = ap.parse_args()

    min_s = _clamp_int(args.sentences_min, 1, 8)
    max_s = _clamp_int(args.sentences_max, min_s, 12)
    max_words = _clamp_int(args.max_words, 4, 60)
    max_panels = _clamp_int(args.max_panels_per_sentence, 1, 6)
    max_ocr = _clamp_int(args.max_ocr_lines, 0, 50)

    in_path = Path(args.storyboard)
    if not in_path.exists():
        _fail(f"File not found: {in_path}")

    doc = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        _fail("Top-level storyboard must be a JSON object.")

    panels = doc.get("panels")
    beats = doc.get("beats")
    if not isinstance(panels, list) or not panels:
        _fail("storyboard.panels must be a non-empty array.")
    if not isinstance(beats, list) or not beats:
        _fail("storyboard.beats must be a non-empty array. Run scripts/make_beats.py first.")

    if isinstance(doc.get("script"), list) and doc.get("script") and not args.overwrite:
        print("script[] already exists; pass --overwrite to replace it.")
        return

    panel_by_id = {}
    for p in panels:
        if isinstance(p, dict) and isinstance(p.get("panel_id"), str):
            panel_by_id[p["panel_id"]] = p

    # Optional provider wiring via existing demo utilities.
    rotator = None
    demo = None
    requests = None
    if args.provider in {"groq", "gemini", "auto", "ollama"}:
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        import requests as _requests  # type: ignore
        import examples.magiv3_demo as _demo  # type: ignore

        requests = _requests
        demo = _demo
        if args.provider in {"gemini", "auto"}:
            keys = demo._load_gemini_keys(args.gemini_key_env)
            if not keys:
                _fail(f"No Gemini keys found in env var {args.gemini_key_env!r} (+ _2.._5).")
            rotator = demo.GeminiKeyRotator(keys, log_key_fingerprint=bool(args.gemini_log_key))
        if args.provider in {"groq"} and not args.groq_model:
            _fail("Missing Groq model. Pass --groq-model or set GROQ_SCRIPT_MODEL in your environment/.env.")

    script_lines: list[dict] = []
    line_idx = 0

    for beat in beats:
        if not isinstance(beat, dict):
            continue
        beat_id = beat.get("beat_id")
        beat_panel_ids = beat.get("panel_ids")
        if not isinstance(beat_id, str) or not isinstance(beat_panel_ids, list) or not beat_panel_ids:
            continue

        # Gather text-only evidence for this beat.
        scene_caps_all: list[str] = []
        scene_tags_all: list[str] = []
        ocr_all: list[str] = []
        for pid in beat_panel_ids:
            panel = panel_by_id.get(pid)
            if not isinstance(panel, dict):
                continue
            caps, tags, ocr = _build_evidence(panel, max_ocr_lines=max_ocr)
            scene_caps_all.extend(caps)
            scene_tags_all.extend(tags)
            ocr_all.extend(ocr)

        # De-dup while preserving order
        def _dedup(xs: list[str]) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for x in xs:
                x = x.strip()
                if not x or x in seen:
                    continue
                out.append(x)
                seen.add(x)
            return out

        scene_caps_all = _dedup(scene_caps_all)[:6]
        scene_tags_all = _dedup(scene_tags_all)[:16]
        ocr_all = _dedup(ocr_all)[:24]

        # Persist compact evidence for debugging/audit in beat.notes (schema-safe).
        beat["notes"] = json.dumps(
            {"scene_captions": scene_caps_all, "scene_tags": scene_tags_all, "ocr_lines": ocr_all},
            ensure_ascii=False,
        )

        if args.provider == "none":
            recaps = _offline_recap_for_beat(beat_panel_ids=beat_panel_ids, scene_caps=scene_caps_all)
        else:
            assert demo is not None and requests is not None

            # Continuity: pass a larger window of previous narrator lines so the paragraph can flow.
            prev_lines = [
                x.get("text", "")
                for x in script_lines[-10:]
                if isinstance(x, dict) and isinstance(x.get("text"), str)
            ]
            prev_lines = [" ".join(x.split()) for x in prev_lines if x and x.strip()]

            # OCR cleanup before extracting story evidence.
            # Keep this conservative: strip SFX/noise, and normalize known OCR name collisions.
            name_map: dict[str, str] = {}
            if "komi" in str(in_path).lower():
                name_map = {"You": "Tadano"}

            ocr_all_clean = _filter_ocr_lines(ocr_all)
            ocr_all_clean = _normalize_problematic_names(ocr_all_clean, name_map)
            ocr_story_lines = _clean_ocr_story_lines(ocr_all_clean, min_words=4)
            ocr_hints = _ocr_to_hints(ocr_story_lines if ocr_story_lines else ocr_all_clean)

            # Optional next-beat hint for smoother transitions (kept minimal and grounded).
            next_summary = ""
            try:
                beat_idx = beats.index(beat)
                if beat_idx + 1 < len(beats) and isinstance(beats[beat_idx + 1], dict):
                    ns = beats[beat_idx + 1].get("summary")
                    if isinstance(ns, str) and ns.strip():
                        next_summary = " ".join(ns.strip().split())
            except Exception:
                next_summary = ""

            prev_block = ""
            if prev_lines:
                prev_block = (
                    "Story so far (for continuity; keep the same flow and vibe, avoid repeating exact wording):\n- "
                    + "\n- ".join(prev_lines)
                    + "\n\n"
                )
            else:
                prev_block = "Story so far (maintain the same flow and vibe; do not repeat exact wording):\n- (none)\n\n"

            # Feed the model cleaned OCR lines as primary evidence; let it extract the actual story facts.
            ocr_facts = _ocr_to_story_facts(ocr_story_lines if ocr_story_lines else ocr_all_clean)

            prompt = (
                "You are a manga recap narrator. Write like a YouTube storyteller — direct, energetic, and easy to follow. "
                "Tell what HAPPENS and why it MATTERS. Never describe what characters look like, how they move, or what expressions they make.\n\n"
                "RULES:\n"
                "- Present tense only.\n"
                "- No dialogue and no quotation marks.\n"
                "- No body language, gestures, or facial expressions.\n"
                "- No opinions or meta-commentary.\n"
                "- Only use facts from the evidence. Do not invent details.\n"
                "- If a character name is also a common English word (like You, Hope, Light), use their last name or role instead.\n"
                "- Ignore all manga sound effects in OCR. SFX are short standalone words like: Beam, Freeze, Crack, Slam, Chew, Stomp, Rustle.\n\n"
                "WRITING STYLE:\n"
                f"- Write ONE paragraph of {min_s} to {max_s} sentences.\n"
                f"- IMPORTANT: Only write as many sentences as the evidence supports. "
                f"If evidence only supports {min_s} sentences, write exactly {min_s}. "
                f"Never invent sentences to reach the maximum.\n"
                f"- Each sentence must be {max_words} words or fewer.\n"
                "- Every sentence must move the story forward.\n"
                "- After the first sentence, connect ideas naturally. Use connectors only when they fit: but, so, then, meanwhile, still, even so.\n"
                "- Vary how sentences start. Never start two sentences in a row with the same word.\n"
                "- No comma inserts or asides.\n"
                "- If the beat ends on tension or a shift, close with one punchy tease sentence.\n\n"
                "EVIDENCE PRIORITY:\n"
                "1. OCR story facts (most important).\n"
                "2. Scene captions (setting/location only).\n"
                "3. Scene tags (ignore unless nothing else is available).\n\n"
                "BEFORE YOU WRITE, do this internally (do not output this):\n"
                "1. Find all character names in the OCR story facts\n"
                "2. Find any decisions, revelations, or if X then Y logic\n"
                "3. Find any emotional stakes or turning points\n"
                "4. Ignore single words and sound effects\n\n"
                "Then write your paragraph using those extracted facts as the foundation.\n"
                "A good sentence reveals what a character thinks, decides, or realizes.\n"
                "A bad sentence only describes what is visible.\n\n"
                "BAD (never write like this):\n"
                "A character sits in a room. Another character looks at them.\n\n"
                "GOOD (always write like this):\n"
                "A character finally tells another character what they have been hiding, and it changes everything.\n\n"
                "The difference: name the characters, state what happens, show why it matters.\n\n"
                "IMPORTANT:\n"
                "- If OCR story facts are present, every sentence must use them as the main source.\n"
                "- Use scene captions only to confirm location, not to replace OCR story facts.\n\n"
                "CONTEXT:\n"
                + prev_block
                + f"Beat panel_ids: {beat_panel_ids}\n"
                + f"Next beat hint: {next_summary or '(none)'}\n\n"
                + "EVIDENCE:\n"
                + ("OCR story facts:\n- " + "\n- ".join(ocr_facts) + "\n\n" if ocr_facts else "OCR story facts: (none)\n\n")
                + (
                    "Scene captions:\n- " + "\n- ".join(scene_caps_all) + "\n\n"
                    if scene_caps_all
                    else "Scene captions: (none)\n\n"
                )
                + ("Scene tags:\n- " + "\n- ".join(scene_tags_all) + "\n\n" if scene_tags_all else "Scene tags: (none)\n\n")
                + "OUTPUT FORMAT:\n"
                + "Return ONLY a single valid JSON object — not an array.\n"
                + "Format: {\"text\": \"FULL PARAGRAPH HERE\", \"panel_ids\": [\"id1\", \"id2\", ...]}\n"
                + "- The text field must contain the entire paragraph as one string.\n"
                + "- panel_ids must only include IDs from this beat.\n"
                + "- Do not include any text outside the JSON object.\n"
            )

            def _gemini_call():
                assert rotator is not None

                def _call(api_key: str):
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{args.gemini_model}:generateContent"
                    payload = {
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
                    }
                    r = requests.post(
                        url, params={"key": api_key}, json=payload, timeout=max(5, int(args.gemini_timeout))
                    )
                    r.raise_for_status()
                    raw = r.json()
                    text = ""
                    try:
                        parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        text = "".join([p.get("text", "") for p in parts if isinstance(p, dict)]).strip()
                    except Exception:
                        text = ""
                    if args.debug_raw:
                        try:
                            payload_dbg = json.loads(beat.get("notes") or "{}") if isinstance(beat.get("notes"), str) else {}
                        except Exception:
                            payload_dbg = {}
                        flat = " ".join(text.split())
                        payload_dbg["provider_raw"] = (flat[:800] + "…") if len(flat) > 800 else flat
                        beat["notes"] = json.dumps(payload_dbg, ensure_ascii=False)
                    return text

                return rotator.call(_call)

            def _ollama_call():
                payload = {
                    "model": args.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 512},
                }
                r = requests.post(
                    f"{args.ollama_host.rstrip('/')}/api/generate",
                    json=payload,
                    timeout=max(5, int(args.ollama_timeout)),
                )
                r.raise_for_status()
                data = r.json()
                text = (data.get("response") or "").strip()
                if args.debug_raw:
                    try:
                        payload_dbg = json.loads(beat.get("notes") or "{}") if isinstance(beat.get("notes"), str) else {}
                    except Exception:
                        payload_dbg = {}
                    flat = " ".join(text.split())
                    payload_dbg["provider_raw"] = (flat[:800] + "…") if len(flat) > 800 else flat
                    beat["notes"] = json.dumps(payload_dbg, ensure_ascii=False)
                return text

            def _groq_call():
                api_key = os.environ.get(args.groq_key_env, "").strip()
                if not api_key:
                    _fail(f"Missing Groq API key env var {args.groq_key_env!r}. Put it in .env or export it.")
                url = f"{args.groq_base_url.rstrip('/')}/chat/completions"
                payload = {
    "model": args.groq_model,
    "messages": [
        {"role": "system", "content": prompt},  # move full prompt here
        {"role": "user", "content": 
            f"Write the recap paragraph for beat {beat_id} using the evidence provided above."},
    ],
    "temperature": 0.7,  # also raise from 0.3 — too low makes output robotic
    "max_tokens": 600,
}
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                r = requests.post(url, headers=headers, json=payload, timeout=max(5, int(args.groq_timeout)))
                r.raise_for_status()
                data = r.json()
                content = (
                    (data.get("choices") or [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                text = (content or "").strip()
                if args.debug_raw:
                    try:
                        payload_dbg = json.loads(beat.get("notes") or "{}") if isinstance(beat.get("notes"), str) else {}
                    except Exception:
                        payload_dbg = {}
                    flat = " ".join(text.split())
                    payload_dbg["provider_raw"] = (flat[:800] + "…") if len(flat) > 800 else flat
                    beat["notes"] = json.dumps(payload_dbg, ensure_ascii=False)
                return text

            def _safe_provider_error(e: Exception) -> str:
                # Avoid leaking secrets in exception messages.
                if requests is not None and isinstance(e, requests.HTTPError):
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    return f"HTTP {status}" if status is not None else "HTTP error"
                if requests is not None and isinstance(e, Exception) and hasattr(requests, "exceptions") and isinstance(
                    e, requests.exceptions.RequestException
                ):
                    return type(e).__name__
                return type(e).__name__

            # Provider selection:
            # - auto: Groq -> Ollama -> Gemini -> offline (never skip)
            # - gemini: Gemini only
            # - ollama: Ollama only
            # - groq: Groq only
            recaps = None  # raw provider text
            err = None
            if args.provider in {"groq", "auto"}:
                try:
                    recaps = _groq_call()
                except Exception as e:
                    if args.provider == "groq":
                        _fail(f"Groq request failed: {_safe_provider_error(e)}")
                    err = e
                    recaps = None
            if recaps is None and args.provider in {"ollama", "auto"}:
                try:
                    recaps = _ollama_call()
                except Exception as e:
                    err = e
                    recaps = None
            if recaps is None and args.provider in {"gemini", "auto"}:
                try:
                    recaps = _gemini_call()
                except Exception as e:
                    err = e
                    recaps = None
            if recaps is None:
                # Provider failed; fall back to offline generation, but record the reason in notes.
                try:
                    payload_dbg = json.loads(beat.get("notes") or "{}") if isinstance(beat.get("notes"), str) else {}
                except Exception:
                    payload_dbg = {}
                payload_dbg["provider_error"] = type(err).__name__ if err else "unknown error"
                beat["notes"] = json.dumps(payload_dbg, ensure_ascii=False)
                recaps = ""

        # Validate + coerce recap items and emit script lines
        items_out: list[dict] = []
        parsed = _parse_beat_output(str(recaps or ""), beat_panel_ids=beat_panel_ids)
        paragraph = parsed.get("paragraph", "")
        pids_all = parsed.get("panel_ids", beat_panel_ids)
        if not isinstance(pids_all, list):
            pids_all = beat_panel_ids
        pids_all = [x for x in pids_all if isinstance(x, str) and x in beat_panel_ids] or beat_panel_ids

        # Split paragraph into sentences, then map panel_ids per sentence.
        para = " ".join(str(paragraph or "").split()).strip()
        if para:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s and s.strip()]
            for sidx, sentence in enumerate(sentences):
                text = _sanitize_sentence(sentence, max_words=max_words)
                if not text:
                    continue
                if _contains_quote_like_apostrophe(text):
                    continue
                if args.reading_level == "simple":
                    text = _simplify_for_simple(text)
                    if _word_count(text) > max_words:
                        text = _trim_to_words(text, max_words)
                if args.reading_level == "kid":
                    text = _simplify_for_kid(text)
                    if _word_count(text) > max_words:
                        text = _trim_to_words(text, max_words)

                pids = _assign_panels_to_sentence(
                    beat_panel_ids=pids_all, sentence_idx=sidx, max_panels_per_sentence=max_panels
                )
                if not pids:
                    pids = beat_panel_ids[:1]
                items_out.append({"text": text, "panel_ids": pids})

        # items_out = _split_to_min_sentences(items_out, min_items=min_s, beat_panel_ids=beat_panel_ids, max_words=max_words)

        if args.provider == "none":
            items_out = _split_to_min_sentences(items_out, min_items=min_s, beat_panel_ids=beat_panel_ids, max_words=max_words)

        # If provider gave too few sentences, add a heuristic bridge sentence before falling back.
        if args.provider == "none" and 0 < len(items_out) < min_s:
            bridge = _bridge_sentence(scene_tags=scene_tags_all, ocr_lines=ocr_all, scene_caps=scene_caps_all, max_words=max_words)
            if bridge:
                # Use next panel in reading order when possible.
                p_bridge = beat_panel_ids[1:2] or beat_panel_ids[:1]
                items_out.append({"text": bridge, "panel_ids": p_bridge})

        # Ensure we never skip: fallback to offline if provider output unusable.
        if len(items_out) < min_s:
            # Offline fallback still tries to feel recap-like, not placeholder-y.
            items_out = _offline_recap_for_beat(beat_panel_ids=beat_panel_ids, scene_caps=scene_caps_all)[:max_s]

        # Clamp to sentences_max
        items_out = items_out[:max_s]

        for it in items_out:
            text = it["text"]
            pids = it["panel_ids"]
            status = "OK"
            if text.startswith("BLOCKED:") or "unclear" in text.lower():
                status = "UNCERTAIN"
            script_lines.append(
                {
                    "line_id": f"l{line_idx:04d}",
                    "beat_id": beat_id,
                    "panel_ids": pids,
                    "text": text,
                    "status": status,
                }
            )
            line_idx += 1

    doc["beats"] = beats
    doc["script"] = script_lines

    out_path = Path(args.out) if args.out else in_path
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"- recap lines: {len(script_lines)}")


if __name__ == "__main__":
    main()
