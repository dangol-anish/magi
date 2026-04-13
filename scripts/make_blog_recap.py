#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _maybe_load_dotenv() -> None:
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
            line = line[len("export "):].lstrip()
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


def _extract_first_json_object(text: str) -> str:
    s = text or ""
    start = s.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start: i + 1]
    return ""


def _extract_first_json_array(text: str) -> str:
    s = text or ""
    start = s.find("[")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start: i + 1]
    return ""


# ── Gemini key rotator ────────────────────────────────────────────────────────

class GeminiKeyRotator:
    ROTATE_ON = {429, 401, 403}
    RETRY_ON = {500, 502, 503, 504}

    def __init__(self, keys: list[str], max_retries: int = 2, base_sleep_s: float = 2.0) -> None:
        if not keys:
            raise ValueError("GeminiKeyRotator requires at least one API key.")
        self.keys = keys
        self.max_retries = int(max_retries)
        self.base_sleep_s = float(base_sleep_s)
        self._idx = 0
        self._exhausted: set[int] = set()

    @property
    def current_key(self) -> str:
        return self.keys[self._idx]

    def _rotate(self) -> bool:
        self._exhausted.add(self._idx)
        for offset in range(1, len(self.keys)):
            candidate = (self._idx + offset) % len(self.keys)
            if candidate not in self._exhausted:
                self._idx = candidate
                return True
        return False

    def call(self, fn):
        import requests
        retries_on_current = 0
        keys_tried = 0
        while True:
            try:
                return fn(self.current_key)
            except requests.HTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in self.ROTATE_ON:
                    keys_tried += 1
                    if keys_tried >= len(self.keys) or not self._rotate():
                        raise RuntimeError(
                            f"All {len(self.keys)} Gemini API key(s) are exhausted or invalid."
                        ) from e
                    retries_on_current = 0
                    continue
                if status in self.RETRY_ON and retries_on_current < self.max_retries:
                    time.sleep(self.base_sleep_s * (2 ** retries_on_current))
                    retries_on_current += 1
                    continue
                raise


def _load_gemini_keys(key_env_var: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()

    def _add(k: str) -> None:
        k = (k or "").strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)

    _add(os.environ.get(key_env_var, ""))
    for n in range(2, 6):
        val = os.environ.get(f"{key_env_var}_{n}", "").strip()
        if not val:
            break
        _add(val)
    return keys


# ── Prompts ───────────────────────────────────────────────────────────────────

# System prompt — Ollama supports this as a separate field, strongly anchors behavior
OLLAMA_SYSTEM_PROMPT = (
    "You are a professional anime and manga recap writer for a popular review blog. "
    "You write in an engaging, vivid, energetic style entirely in your own words. "
    "You NEVER copy or echo the input text — you always transform raw notes into "
    "polished prose. You follow formatting instructions exactly, complete every part "
    "of the task, and when asked for JSON you output only valid JSON with no markdown "
    "fences, no preamble, and no trailing commentary."
)

# Main prompt — uses .replace("{raw_script}", ...) so literal braces in JSON examples are safe
PROMPT_TEMPLATE = """\
You are rewriting rough OCR panel notes from a manga into a polished chapter recap.

CRITICAL RULES — read carefully before starting:
1. Do NOT copy the input. Every sentence must be rewritten fresh in your own words.
2. The input has 6 pages (0-5) with a lot of repeated content — collapse it all into exactly 3-4 paragraphs.
3. Correct all character details from the OCR errors:
   - The red-haired wanderer = Himura Kenshin (he/him) — a legendary swordsman hiding his past
   - The woman who confronts him = Kamiya Kaoru — a martial artist and dojo instructor
   - "Battousai" / "Battou-Sai" = Battosai — Kenshin's feared assassin alias from the revolution
   - "Oro" = Kenshin's goofy catchphrase of surprise or confusion
   - Kenshin is MALE — fix any gender errors from the OCR
4. Write in the style of an enthusiastic anime review blog — vivid, punchy, with a sense of drama and humor
5. Do not spoil anything beyond what happens in this chapter

Once you have written the recap, output the result as a single JSON object.
No markdown fences. No explanation before or after. Start with {{ and end with }}.

{{
  "recap": "<your full recap with paragraphs separated by \\n\\n>",
  "segment_map": [
    {{ "paragraph": 1, "page_indices": [0, 1] }},
    {{ "paragraph": 2, "page_indices": [2, 3] }},
    {{ "paragraph": 3, "page_indices": [4, 5] }}
  ]
}}

The segment_map above is a suggested starting point — adjust page_indices to match what you wrote.
Every paragraph must appear in the segment_map. Do not skip any.

RAW PANEL NOTES (rewrite these — do NOT copy them):
{raw_script}
"""

# Stricter retry prompt after a JSON parse failure
RETRY_PROMPT_TEMPLATE = """\
Your previous response did not contain valid JSON. Try again from scratch.

Output ONLY a single JSON object. No markdown fences, no explanation, no text before {{ or after }}.

{{
  "recap": "<3-4 paragraph recap written in your own words, paragraphs joined with \\n\\n>",
  "segment_map": [
    {{ "paragraph": 1, "page_indices": [0, 1] }},
    {{ "paragraph": 2, "page_indices": [2, 3] }},
    {{ "paragraph": 3, "page_indices": [4, 5] }}
  ]
}}

RAW PANEL NOTES (for reference — rewrite, do not copy):
{raw_script}
"""


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty output")

    # Strip markdown fences if the model added them anyway
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

    # Try strict parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Extract first balanced JSON object from noisy surrounding text
    blob = _extract_first_json_object(text) or _extract_first_json_array(text) or ""
    if not blob:
        raise ValueError("no JSON object found in output")

    parsed2 = json.loads(blob)
    if isinstance(parsed2, dict):
        return parsed2

    # Bare array fallback — treat as segment_map
    if isinstance(parsed2, list) and parsed2 and all(isinstance(x, dict) for x in parsed2):
        return {"recap": "", "segment_map": parsed2}

    raise ValueError("JSON parsed but is not an object")


# ── Provider calls ────────────────────────────────────────────────────────────

def _safe_provider_error(e: Exception) -> str:
    try:
        import requests

        if isinstance(e, requests.HTTPError):
            status = getattr(getattr(e, "response", None), "status_code", None)
            return f"HTTP {status}" if status is not None else "HTTP error"
        if hasattr(requests, "exceptions") and isinstance(e, requests.exceptions.RequestException):
            return type(e).__name__
    except Exception:
        pass
    return type(e).__name__


def _call_ollama(
    prompt: str,
    model: str,
    host: str,
    temperature: float,
    num_ctx: int,
    timeout: int,
    progress: bool,
) -> str:
    import requests

    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "system": OLLAMA_SYSTEM_PROMPT,   # <-- strongly anchors behavior
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": -1,             # <-- unlimited output, no token cap
            "num_ctx": num_ctx,
        },
    }

    if progress:
        print(f"[BlogRecap] ollama POST {url}  model={model}  ctx={num_ctx}", flush=True)

    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def _call_groq(
    prompt: str,
    *,
    model: str,
    key_env_var: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    progress: bool,
) -> str:
    import requests

    api_key = (os.environ.get(key_env_var, "") or "").strip()
    if not api_key and str(key_env_var) == "GROQ_SCRIPT_KEY":
        api_key = (os.environ.get("GROQ_API_KEY", "") or "").strip() or (os.environ.get("GROQ_KEY", "") or "").strip()
    if not api_key:
        _fail(
            f"Missing Groq API key env var {key_env_var!r}. "
            "If your key is stored in GROQ_API_KEY, re-run with: --groq-key-env GROQ_API_KEY"
        )

    url = f"{str(base_url).rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": str(model),
        "messages": [
            {"role": "system", "content": "Output ONLY valid JSON (no markdown, no extra text)."},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max(256, max_tokens)),
    }

    if progress:
        print(f"[BlogRecap] groq POST {url}  model={model}", flush=True)

    r = requests.post(url, headers=headers, json=payload, timeout=max(5, int(timeout)))
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # Bubble up a compact, helpful error (no secrets).
        msg = ""
        try:
            data = r.json()
            err = data.get("error") if isinstance(data, dict) else None
            if isinstance(err, dict):
                em = err.get("message")
                es = err.get("type") or err.get("code") or err.get("status")
                if em and es:
                    msg = f"{es}: {em}"
                elif em:
                    msg = str(em)
        except Exception:
            msg = ""
        if msg:
            raise RuntimeError(msg) from e
        raise
    data = r.json()
    try:
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except Exception:
        return ""


def _call_gemini(
    prompt: str,
    model: str,
    key_env_var: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    progress: bool,
) -> str:
    import requests

    keys = _load_gemini_keys(key_env_var)
    if not keys:
        _fail(f"No Gemini keys found in env var {key_env_var!r} (+ _2.._5).")
    rotator = GeminiKeyRotator(keys)

    def _extract_text(raw: dict) -> str:
        parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()

    def _post(api_key: str, mdl: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        r = requests.post(url, params={"key": api_key}, json=payload, timeout=max(5, timeout))
        r.raise_for_status()
        return _extract_text(r.json())

    requested = model.strip()
    fallbacks = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    tried: list[str] = []

    def _try(mdl: str) -> str | None:
        tried.append(mdl)
        try:
            return rotator.call(lambda k: _post(k, mdl))
        except Exception as e:
            if isinstance(e, requests.HTTPError):
                if getattr(getattr(e, "response", None), "status_code", None) == 404:
                    return None
            raise

    text = _try(requested)
    if text is not None:
        if progress:
            print(f"[BlogRecap] used model: {requested}", flush=True)
        return text

    for m in fallbacks:
        if m == requested:
            continue
        text = _try(m)
        if text is not None:
            if progress:
                print(f"[BlogRecap] fell back from {requested} to {m}", flush=True)
            return text

    _fail(f"Gemini model not found (404). Tried: {', '.join(tried)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _maybe_load_dotenv()

    ap = argparse.ArgumentParser(
        description="Stage 4 (blog recap): rewrite recap_pages.json into a cohesive blog recap + paragraph->page mapping."
    )
    ap.add_argument("recap_pages_json", help="Path to final/recap_pages.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--debug-raw", action="store_true")

    ap.add_argument("--provider", choices=["gemini", "ollama", "groq"], default="gemini",
                    help="Which provider to use (default: gemini)")

    # Gemini options
    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    ap.add_argument("--gemini-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--gemini-timeout", type=int, default=180)
    ap.add_argument("--max-output-tokens", type=int, default=4000)  # raised from 1200

    # Groq options (OpenAI-compatible)
    ap.add_argument("--groq-key-env", default="GROQ_SCRIPT_KEY")
    ap.add_argument("--groq-model", default=os.environ.get("GROQ_SCRIPT_MODEL", ""))
    ap.add_argument("--groq-base-url", default=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    ap.add_argument("--groq-timeout", type=int, default=120)
    ap.add_argument("--groq-max-tokens", type=int, default=1200)

    # Ollama options
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", default="qwen2.5:7b")
    ap.add_argument("--ollama-timeout", type=int, default=300)
    ap.add_argument("--num-ctx", type=int, default=8192)

    ap.add_argument("--temperature", type=float, default=0.7,   # raised from 0.3 — helps break copy mode
                    help="Sampling temperature (default: 0.7)")
    ap.add_argument("--max-raw-chars", type=int, default=60000)
    ap.add_argument("--max-attempts", type=int, default=2)

    args = ap.parse_args()

    # ── Load input ────────────────────────────────────────────────────────────
    in_path = Path(args.recap_pages_json)
    if not in_path.exists():
        _fail(f"File not found: {in_path}")

    try:
        doc = json.loads(in_path.read_text(encoding="utf-8"))
    except Exception as e:
        _fail(f"Failed to parse JSON: {e}")

    if not isinstance(doc, dict):
        _fail("Input must be a JSON object.")

    raw_script = doc.get("raw_script")
    if not isinstance(raw_script, str) or not raw_script.strip():
        _fail("Input JSON missing non-empty 'raw_script'.")
    raw_script = raw_script.strip()

    if len(raw_script) > args.max_raw_chars:
        raw_script = raw_script[:args.max_raw_chars].rstrip() + "\n\n[TRUNCATED]"

    # ── Output path ───────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else (in_path.parent / "recap_blog.json")
    if out_path.exists() and not args.overwrite:
        print(f"Output exists (skip): {out_path}")
        print("Pass --overwrite to regenerate.")
        return

    if args.progress:
        if args.provider == "gemini":
            model = args.gemini_model
        elif args.provider == "groq":
            model = (args.groq_model or os.environ.get("GROQ_SCRIPT_MODEL", "") or os.environ.get("GROQ_MODEL", "")).strip()
        else:
            model = args.ollama_model
        print(
            f"[BlogRecap] provider={args.provider}  model={model}  "
            f"in={in_path.name}  out={out_path.name}",
            flush=True,
        )

    # ── Inference loop ────────────────────────────────────────────────────────
    active_prompt = PROMPT_TEMPLATE.replace("{raw_script}", raw_script)

    raw_text = ""
    last_err: Exception | None = None
    max_attempts = max(1, args.max_attempts)

    for attempt in range(1, max_attempts + 1):
        temperature = args.temperature if attempt == 1 else 0.0

        t0 = time.time()
        try:
            if args.provider == "ollama":
                raw_text = _call_ollama(
                    prompt=active_prompt,
                    model=args.ollama_model,
                    host=args.ollama_host,
                    temperature=temperature,
                    num_ctx=args.num_ctx,
                    timeout=args.ollama_timeout,
                    progress=args.progress,
                )
            elif args.provider == "groq":
                model = (args.groq_model or os.environ.get("GROQ_SCRIPT_MODEL", "") or os.environ.get("GROQ_MODEL", "")).strip()
                if not model:
                    _fail("Missing Groq model. Pass --groq-model or set GROQ_SCRIPT_MODEL.")
                raw_text = _call_groq(
                    active_prompt,
                    model=model,
                    key_env_var=args.groq_key_env,
                    base_url=args.groq_base_url,
                    temperature=temperature,
                    max_tokens=args.groq_max_tokens,
                    timeout=args.groq_timeout,
                    progress=args.progress,
                )
            else:
                raw_text = _call_gemini(
                    prompt=active_prompt,
                    model=args.gemini_model,
                    key_env_var=args.gemini_key_env,
                    temperature=temperature,
                    max_output_tokens=args.max_output_tokens,
                    timeout=args.gemini_timeout,
                    progress=args.progress,
                )
        except Exception as e:
            _fail(f"Provider request failed: {_safe_provider_error(e)}")

        if args.progress:
            print(f"[BlogRecap] response in {time.time() - t0:.1f}s (attempt {attempt}/{max_attempts})", flush=True)

        try:
            out_obj = _extract_json(raw_text)
            last_err = None
            break
        except Exception as e:
            last_err = e
            if args.progress:
                print(f"[BlogRecap] JSON parse failed (attempt {attempt}): {e}", flush=True)
            if attempt >= max_attempts:
                break
            # Switch to stricter retry prompt
            active_prompt = RETRY_PROMPT_TEMPLATE.replace("{raw_script}", raw_script)

    if last_err is not None:
        if args.debug_raw:
            raw_path = out_path.with_suffix(out_path.suffix + ".raw.txt")
            try:
                raw_path.write_text(raw_text or "", encoding="utf-8")
                print(f"[BlogRecap] raw output -> {raw_path}", flush=True)
            except Exception:
                pass
        _fail(
            f"Provider did not return valid JSON after {max_attempts} attempt(s). "
            f"Run with --debug-raw to inspect. "
            f"Try --ollama-model qwen2.5:14b for better reliability."
        )

    # ── Validate output ───────────────────────────────────────────────────────
    recap = out_obj.get("recap")
    segment_map = out_obj.get("segment_map")

    if not isinstance(recap, str) or not recap.strip():
        _fail("Provider output missing non-empty 'recap' string.")
    if not isinstance(segment_map, list) or not segment_map:
        _fail("Provider output missing non-empty 'segment_map' array.")

    seg_out: list[dict] = []
    for item in segment_map:
        if not isinstance(item, dict):
            continue
        try:
            para_i = int(item.get("paragraph"))
        except Exception:
            continue
        pages: list[int] = []
        for x in (item.get("page_indices") or []):
            try:
                pages.append(int(x))
            except Exception:
                continue
        seg_out.append({"paragraph": para_i, "page_indices": sorted(set(pages))})

    if not seg_out:
        _fail("Provider output had no usable segment_map entries.")

    seg_out = sorted(seg_out, key=lambda x: int(x.get("paragraph") or 0))

    # ── Write output ──────────────────────────────────────────────────────────
    out_path.write_text(
        json.dumps({"recap": recap.strip(), "segment_map": seg_out}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.progress:
        print(f"[BlogRecap] wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
