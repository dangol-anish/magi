import argparse
import os
import json
import base64
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
import requests


def _ensure_hf_cache_writable() -> None:
    # Hugging Face caches to ~/.cache by default. In some environments (sandboxes, CI, shared accounts),
    # that path may not be writable. If the user hasn't set HF_HOME, default it to a project-local dir.
    if os.environ.get("HF_HOME"):
        return
    project_root = Path(__file__).resolve().parents[1]
    hf_home = project_root / ".hf"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)


_ensure_hf_cache_writable()


from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: E402


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(device: str) -> torch.dtype:
    # macOS typically has no CUDA; float16 helps on MPS but is usually unsupported/slow on CPU.
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def _read_image_rgb_np(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        image = Image.open(fh).convert("RGB")
    return np.array(image)


def _collect_image_paths(raw_inputs: list[str]) -> list[str]:
    supported_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    missing: list[str] = []
    resolved: list[str] = []

    for raw in raw_inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in supported_suffixes:
                    resolved.append(str(child))
            continue

        if p.is_file():
            resolved.append(str(p))
            continue

        missing.append(str(p))

    if missing:
        hint = (
            "Pass real image paths, e.g.\n"
            "  python examples/magiv3_demo.py --images ~/Downloads/page1.png\n"
            "or pass a folder:\n"
            "  python examples/magiv3_demo.py --images ~/Downloads/chapter_pages/\n"
        )
        raise SystemExit(f"Image path(s) not found:\n- " + "\n- ".join(missing) + "\n\n" + hint)

    if not resolved:
        raise SystemExit("No images found. Provide image files or a folder containing images.")

    return resolved


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _maybe_extract_polygon(bbox):
    """
    Try to interpret `bbox` as either:
      - [x1, y1, x2, y2]
      - [[x, y], [x, y], ...]
      - {"bbox": ...} / {"box": ...} / {"polygon": ...}
    Returns (polygon_points, rect_xyxy) where either may be None.
    """
    if bbox is None:
        return None, None

    if isinstance(bbox, dict):
        for key in ("polygon", "poly", "points", "segmentation"):
            if key in bbox:
                return _maybe_extract_polygon(bbox[key])
        for key in ("bbox", "box", "rect"):
            if key in bbox:
                return _maybe_extract_polygon(bbox[key])
        return None, None

    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(x, (int, float, np.integer, np.floating)) for x in bbox):
        x1, y1, x2, y2 = [float(x) for x in bbox]
        return None, (x1, y1, x2, y2)

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 3 and all(
        isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(x, (int, float, np.integer, np.floating)) for x in p)
        for p in bbox
    ):
        pts = [(float(x), float(y)) for x, y in bbox]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return pts, (min(xs), min(ys), max(xs), max(ys))

    return None, None


def _draw_overlay(image_path: str, result: dict, out_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Panels (yellow)
    for panel in result.get("panels", []) or []:
        poly, rect = _maybe_extract_polygon(panel)
        if poly:
            draw.polygon(poly, outline=(255, 215, 0, 255))
        elif rect:
            draw.rectangle(rect, outline=(255, 215, 0, 255), width=3)

    # Texts (red), fill lightly if essential
    essential = result.get("is_essential_text", []) or []
    texts = result.get("texts", []) or []
    for idx, tb in enumerate(texts):
        poly, rect = _maybe_extract_polygon(tb)
        is_ess = bool(essential[idx]) if idx < len(essential) else False
        fill = (255, 0, 0, 40) if is_ess else None
        if poly:
            if fill:
                draw.polygon(poly, fill=fill, outline=(255, 0, 0, 255))
            else:
                draw.polygon(poly, outline=(255, 0, 0, 255))
        elif rect:
            if fill:
                draw.rectangle(rect, fill=fill, outline=(255, 0, 0, 255), width=3)
            else:
                draw.rectangle(rect, outline=(255, 0, 0, 255), width=3)

    # Characters (blue)
    for cb in result.get("characters", []) or []:
        poly, rect = _maybe_extract_polygon(cb)
        if poly:
            draw.polygon(poly, outline=(0, 120, 255, 255))
        elif rect:
            draw.rectangle(rect, outline=(0, 120, 255, 255), width=3)

    img.save(out_path)


def _extract_ocr_texts(ocr_page):
    """
    Magi v3 returns OCR in a few possible shapes depending on remote code revisions.
    Normalize into a list[str] if we can.
    """
    if ocr_page is None:
        return []
    if isinstance(ocr_page, list):
        # Sometimes a list[str], sometimes list[dict]
        out: list[str] = []
        for x in ocr_page:
            if x is None:
                out.append("")
                continue
            if isinstance(x, str):
                out.append(x)
                continue
            if isinstance(x, dict):
                for key in ("text", "ocr", "prediction", "pred", "value"):
                    if key in x:
                        out.append("" if x[key] is None else str(x[key]))
                        break
                else:
                    out.append(str(x))
                continue
            out.append(str(x))
        return out
    if isinstance(ocr_page, dict):
        for key in ("ocr_texts", "ocr", "texts", "text", "predictions"):
            if key in ocr_page and isinstance(ocr_page[key], list):
                return [str(x) if x is not None else "" for x in ocr_page[key]]
    return [str(ocr_page)]


def _build_transcript(result: dict, ocr_texts: list[str]) -> tuple[list[dict], str]:
    essential = result.get("is_essential_text", []) or []
    texts = result.get("texts", []) or []
    assoc = result.get("text_character_associations", []) or []
    char_labels = result.get("character_cluster_labels", []) or []

    # text_idx -> char_idx
    text_to_char: dict[int, int] = {}
    if isinstance(assoc, list):
        for pair in assoc:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                try:
                    text_to_char[int(pair[0])] = int(pair[1])
                except Exception:
                    continue

    items: list[dict] = []
    lines: list[str] = []
    n = max(len(texts), len(ocr_texts), len(essential))
    for text_idx in range(n):
        is_ess = bool(essential[text_idx]) if text_idx < len(essential) else True
        if not is_ess:
            continue

        raw_text = ocr_texts[text_idx] if text_idx < len(ocr_texts) else ""
        char_idx = text_to_char.get(text_idx)
        speaker = "unsure"
        if char_idx is not None:
            if 0 <= char_idx < len(char_labels):
                speaker = f"char_{char_labels[char_idx]}"
            else:
                speaker = f"char_{char_idx}"

        bbox = texts[text_idx] if text_idx < len(texts) else None
        items.append(
            {
                "text_idx": text_idx,
                "speaker": speaker,
                "text": raw_text,
                "bbox": bbox,
                "essential": True,
            }
        )
        lines.append(f"<{speaker}>: {raw_text}")

    return items, "\n".join(lines) + ("\n" if lines else "")


def _generate_text(
    model,
    processor,
    images: list[np.ndarray],
    input_texts: list[str],
    input_bboxes: list[list[list[list[float]]]] | None = None,
    max_new_tokens: int = 128,
    num_beams: int = 1,
    suppress_structure_tokens: bool = False,
):
    if input_bboxes is None:
        input_bboxes = [[] for _ in range(len(input_texts))]
    batch_inputs = processor(
        batch_input_text=input_texts,
        batch_input_list_of_list_of_bboxes=input_bboxes,
        batch_images=images,
        padding=True,
        truncation=True,
        max_input_length_including_image_tokens=1024,
        max_output_length=min(1024, max_new_tokens + 64),
        return_tensors="pt",
        dtype=model.dtype,
        device=model.device,
    )
    generated_ids = model.generate(
        input_ids=batch_inputs["input_ids"],
        pixel_values=batch_inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
        early_stopping=False if num_beams == 1 else True,
        bad_words_ids=_build_bad_words_ids(processor) if suppress_structure_tokens else None,
    )
    generated_texts, _, _ = processor.postprocess_output(generated_ids, images)
    return [_cleanup_natural_text(t) for t in generated_texts]


_BAD_WORDS_IDS = None


def _build_bad_words_ids(processor):
    """
    For caption/prose, Magi v3 sometimes emits structural tokens like <panel> or <loc_123>.
    Suppressing them nudges generation toward natural language.
    """
    global _BAD_WORDS_IDS
    if _BAD_WORDS_IDS is not None:
        return _BAD_WORDS_IDS

    tokenizer = processor.tokenizer
    bad_token_ids: list[list[int]] = []

    # Common structure tokens used by this model family
    for tok in [
        "<panel>",
        "<text>",
        "<character>",
        "<tail>",
        "<od>",
        "</od>",
        "<ocr>",
        "</ocr>",
        "<cap>",
        "</cap>",
        "<ncap>",
        "</ncap>",
        "<dcap>",
        "</dcap>",
        "<region_cap>",
        "</region_cap>",
        "<region_to_desciption>",
        "</region_to_desciption>",
        "<proposal>",
        "</proposal>",
        "<poly>",
        "</poly>",
        "<and>",
        "<sep>",
        "<seg>",
        "</seg>",
        "<grounding>",
        "</grounding>",
    ]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            bad_token_ids.append([tid])

    # Location tokens
    for i in range(1000):
        tok = f"<loc_{i}>"
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            bad_token_ids.append([tid])

    _BAD_WORDS_IDS = bad_token_ids
    return _BAD_WORDS_IDS


def _cleanup_natural_text(text: str) -> str:
    # Strip any remaining <...> tags and normalize whitespace.
    import re

    t = re.sub(r"<[^>]+>", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _safe_rect_xyxy(bbox):
    _, rect = _maybe_extract_polygon(bbox)
    return rect


def _clamp_rect(rect, width: int, height: int):
    x1, y1, x2, y2 = rect
    x1 = max(0.0, min(float(width), float(x1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y1 = max(0.0, min(float(height), float(y1)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _crop_rect(image: Image.Image, rect, pad_ratio: float = 0.03) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = _clamp_rect(rect, w, h)
    dx = (x2 - x1) * pad_ratio
    dy = (y2 - y1) * pad_ratio
    x1 = max(0, int(x1 - dx))
    y1 = max(0, int(y1 - dy))
    x2 = min(w, int(x2 + dx))
    y2 = min(h, int(y2 + dy))
    if x2 <= x1 or y2 <= y1:
        return image.copy()
    return image.crop((x1, y1, x2, y2))


def _ollama_generate_text(
    *,
    host: str,
    model: str,
    prompt: str,
    image: Image.Image | None = None,
    max_tokens: int = 128,
    temperature: float = 0.2,
    timeout_s: int = 120,
):
    """
    Calls local Ollama (http://127.0.0.1:11434) with a single image.
    Returns the parsed JSON object if possible; otherwise returns {"raw": "..."}.
    """
    images = None
    if image is not None:
        import io

        bio = io.BytesIO()
        image.save(bio, format="PNG")
        img_b64 = base64.b64encode(bio.getvalue()).decode("ascii")
        images = [img_b64]

    payload = {
        "model": model,
        "prompt": prompt,
        **({"images": images} if images is not None else {}),
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": int(max_tokens),
        },
    }
    r = requests.post(f"{host.rstrip('/')}/api/generate", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def _ollama_caption_prompt() -> str:
    return (
        "Describe the visual scene in this manga panel in ONE short sentence.\n"
        "Ignore any written dialogue text in speech bubbles.\n"
    )


def _ollama_tags_prompt(caption: str) -> str:
    return (
        "Extract 2 to 6 short lowercase scene tags from this caption.\n"
        "Return exactly one line:\n"
        "TAGS: tag1, tag2, tag3\n"
        f"Caption: {caption}\n"
    )


def _parse_scene_labels(text: str) -> tuple[list[str], str]:
    if not text:
        return [], ""
    tags: list[str] = []
    caption = ""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("tags:"):
            raw = s.split(":", 1)[1].strip()
            parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
            tags = [p for p in parts if p not in {"...", "tag", "tags"}]
        elif s.lower().startswith("caption:"):
            caption = s.split(":", 1)[1].strip()
    if not caption:
        # fallback: use whole text as caption
        caption = " ".join([ln.strip() for ln in text.splitlines() if ln.strip()])[:400]
    return tags[:6], caption


def _strip_json_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        # remove code fences if present
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _extract_first_json_object(text: str) -> str | None:
    """
    Best-effort extraction of the first JSON object in a string.
    """
    t = _strip_json_fence(text)
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]
    return None


def _extract_first_json_array(text: str) -> str | None:
    """
    Best-effort extraction of the first JSON array in a string.
    """
    t = _strip_json_fence(text)
    start = t.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]
    return None


def _sha256_png(image: Image.Image) -> str:
    import io

    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return hashlib.sha256(bio.getvalue()).hexdigest()


def _gemini_scene_prompt() -> str:
    return (
        "Label manga panels. Ignore written dialogue; focus on visuals.\n"
        "Return ONLY JSON array:\n"
        "[{\"panel_idx\":0,\"tags\":[\"tag\"],\"caption\":\"one short sentence\"}]\n"
        "tags: 2-6 short lowercase labels."
    )


def _gemini_generate_scene_json_batch(
    *,
    api_key: str,
    model: str,
    panel_images: list[tuple[int, Image.Image]],
    max_tokens: int,
    temperature: float,
    thinking_budget: int,
    timeout_s: int,
) -> tuple[list[dict], dict]:
    """
    Calls Google Gemini (Generative Language API) to get {tags, caption} for multiple panels in one request.
    Returns (items, raw_response).
    """
    import io

    parts = [{"text": _gemini_scene_prompt()}]
    for panel_idx, im in panel_images:
        # include a tiny text marker before each image so the model can align outputs
        parts.append({"text": f"PANEL {panel_idx}:"})
        bio = io.BytesIO()
        im.save(bio, format="PNG")
        img_b64 = base64.b64encode(bio.getvalue()).decode("ascii")
        parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
            # Gemini 2.5 models have thinking enabled by default. Turn it off for lower latency/quota.
            # (If the model doesn't support it, it should be ignored.)
            "thinkingConfig": {"thinkingBudget": int(thinking_budget)},
        },
    }
    r = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout_s)
    r.raise_for_status()
    raw = r.json()

    text = ""
    try:
        parts = raw["candidates"][0]["content"]["parts"]
        text = "".join([p.get("text", "") for p in parts if isinstance(p, dict)]).strip()
    except Exception:
        text = ""

    items: list[dict] = []
    parsed = None
    obj_text = _extract_first_json_array(text) or _extract_first_json_object(text) or ""
    if obj_text:
        try:
            parsed = json.loads(obj_text)
        except Exception:
            parsed = None

    if isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            panel_idx = entry.get("panel_idx")
            tags = entry.get("tags") if isinstance(entry.get("tags"), list) else []
            caption = entry.get("caption") if isinstance(entry.get("caption"), str) else ""
            try:
                panel_idx = int(panel_idx)
            except Exception:
                continue
            tags = [str(x).strip().lower() for x in tags if str(x).strip()]
            tags = [t for t in tags if t and t not in {"...", "tag", "tags"}][:6]
            items.append({"panel_idx": panel_idx, "tags": tags, "caption": caption.strip()})
    elif isinstance(parsed, dict):
        # Single object fallback (treat as panel_idx of the first provided image)
        first_idx = panel_images[0][0] if panel_images else 0
        tags = parsed.get("tags") if isinstance(parsed.get("tags"), list) else []
        caption = parsed.get("caption") if isinstance(parsed.get("caption"), str) else ""
        tags = [str(x).strip().lower() for x in tags if str(x).strip()]
        tags = [t for t in tags if t and t not in {"...", "tag", "tags"}][:6]
        items.append({"panel_idx": first_idx, "tags": tags, "caption": caption.strip()})

    finish_reason = None
    try:
        finish_reason = raw.get("candidates", [{}])[0].get("finishReason")
    except Exception:
        finish_reason = None

    usage = raw.get("usageMetadata") if isinstance(raw, dict) else None
    return items, {"response_text": text, "finishReason": finish_reason, "usageMetadata": usage, "raw": raw}


def _gemini_post_with_backoff(fn, max_retries: int = 2, base_sleep_s: float = 2.0):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            # On quota/overload, don't hammer; back off and retry a little, then give up.
            if status in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(base_sleep_s * (2**attempt))
                continue
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Magi v3 demo runner (macOS-friendly).")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="One or more input images (png/jpg).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device. Use 'mps' on Apple Silicon if available.",
    )
    parser.add_argument(
        "--model",
        default="ragavsachdeva/magiv3",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--attn",
        default="eager",
        choices=["auto", "eager", "sdpa"],
        help="Attention implementation. 'eager' is the safest default for remote-code models.",
    )
    parser.add_argument(
        "--out",
        default="out",
        help="Output directory (JSON, transcript, and annotated images).",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip saving annotated images.",
    )
    parser.add_argument(
        "--panel-captions",
        action="store_true",
        help="Generate a caption for each detected panel.",
    )
    parser.add_argument(
        "--panel-grounding",
        action="store_true",
        help="Run character grounding on each panel caption (implies --panel-captions).",
    )
    parser.add_argument(
        "--max-panels",
        type=int,
        default=6,
        help="Max number of panels to caption/ground per page.",
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=1,
        help="Beam search width for captioning/prose (lower is faster; 1 is fastest).",
    )
    parser.add_argument(
        "--caption-tokens",
        type=int,
        default=64,
        help="Max new tokens for each panel caption (lower is faster).",
    )
    parser.add_argument(
        "--prose-tokens",
        type=int,
        default=160,
        help="Max new tokens for page-level prose (lower is faster).",
    )
    parser.add_argument(
        "--prose",
        action="store_true",
        help="Generate a short prose paragraph for the full page.",
    )
    parser.add_argument(
        "--ollama-scene-labels",
        action="store_true",
        help="Use local Ollama to generate scene tags + a short caption for each detected panel crop.",
    )
    parser.add_argument(
        "--scene-provider",
        choices=["ollama", "gemini", "auto"],
        default="ollama",
        help="Provider for panel scene labels.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://127.0.0.1:11434",
        help="Ollama host URL.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llava-phi3:latest",
        help="Ollama vision model to use for scene labels (e.g. llava-phi3:latest, moondream:latest, qwen2.5vl:3b).",
    )
    parser.add_argument(
        "--scene-tokens",
        type=int,
        default=96,
        help="Max tokens for each Ollama scene label response (lower is faster).",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.0-flash",
        help="Gemini model name (e.g. gemini-2.0-flash).",
    )
    parser.add_argument(
        "--gemini-key-env",
        default="GEMINI_API_KEY",
        help="Env var name containing your Gemini API key.",
    )
    parser.add_argument(
        "--gemini-timeout",
        type=int,
        default=120,
        help="Gemini request timeout in seconds.",
    )
    parser.add_argument(
        "--gemini-batch-size",
        type=int,
        default=6,
        help="How many panel crops to send per Gemini request (lower reduces request size; higher reduces total calls).",
    )
    parser.add_argument(
        "--gemini-thinking-budget",
        type=int,
        default=0,
        help="Gemini 2.5 thinking budget. Use 0 for cheapest/fastest.",
    )
    args = parser.parse_args()

    device = _pick_device(args.device)
    dtype = _dtype_for_device(device)

    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("Requested --device mps, but MPS is not available on this machine.")

    print(f"Loading model={args.model!r} on device={device!r} dtype={str(dtype).replace('torch.', '')}...")

    attn_implementation = None if args.attn == "auto" else args.attn

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    image_paths = _collect_image_paths(args.images)
    images = [_read_image_rgb_np(p) for p in image_paths]

    with torch.no_grad():
        det_assoc = model.predict_detections_and_associations(images, processor)
        ocr = model.predict_ocr(images, processor)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript_items_all: list[dict] = []
    transcript_lines_all: list[str] = []

    for page_idx, (image_path, page_result, page_ocr) in enumerate(zip(image_paths, det_assoc, ocr)):
        page_result = page_result if isinstance(page_result, dict) else {"result": page_result}
        ocr_texts = _extract_ocr_texts(page_ocr)

        # JSON dump
        page_out = {
            "image_path": image_path,
            "detections_and_associations": page_result,
            "ocr_raw": page_ocr,
            "ocr_texts": ocr_texts,
        }
        (out_dir / f"page_{page_idx}.json").write_text(
            json.dumps(_to_jsonable(page_out), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Transcript
        transcript_items, transcript_txt = _build_transcript(page_result, ocr_texts)
        for item in transcript_items:
            item["page_idx"] = page_idx
            transcript_items_all.append(item)
        if transcript_txt:
            transcript_lines_all.append(f"# page {page_idx}")
            transcript_lines_all.append(transcript_txt.rstrip("\n"))

        # Visualization
        if not args.no_viz and isinstance(page_result, dict):
            _draw_overlay(image_path, page_result, str(out_dir / f"page_{page_idx}_annotated.png"))

        # Scene labels (Ollama or Gemini)
        if args.ollama_scene_labels and isinstance(page_result, dict):
            pil_page = Image.open(image_path).convert("RGB")
            panel_rects = []
            for pb in (page_result.get("panels", []) or []):
                rect = _safe_rect_xyxy(pb)
                if rect is None:
                    continue
                panel_rects.append(rect)
            panel_rects = sorted(panel_rects, key=lambda r: (float(r[1]), float(r[0])))
            panel_rects = panel_rects[: max(0, args.max_panels)]

            scene_items: list[dict] = []

            if args.scene_provider == "ollama":
                caption_prompt = _ollama_caption_prompt()
                for panel_idx, rect in enumerate(panel_rects):
                    crop = _crop_rect(pil_page, rect)
                    try:
                        caption_text = _ollama_generate_text(
                            host=args.ollama_host,
                            model=args.ollama_model,
                            prompt=caption_prompt,
                            image=crop,
                            max_tokens=max(16, args.scene_tokens),
                        )
                    except Exception as e:
                        caption_text = f"ERROR: {e}"

                    try:
                        tags_text = _ollama_generate_text(
                            host=args.ollama_host,
                            model=args.ollama_model,
                            prompt=_ollama_tags_prompt(caption_text),
                            image=None,
                            max_tokens=64,
                            temperature=0.0,
                        )
                    except Exception as e:
                        tags_text = f"ERROR: {e}"

                    tags, _ = _parse_scene_labels(tags_text)
                    scene_items.append(
                        {
                            "page_idx": page_idx,
                            "panel_idx": panel_idx,
                            "bbox": list(rect),
                            "tags": tags,
                            "caption": caption_text.strip(),
                            "raw": {"provider": "ollama", "caption_raw": caption_text, "tags_raw": tags_text},
                        }
                    )

            else:
                # Gemini: batch panels per page + cache so reruns don't spend quota.
                api_key = os.environ.get(args.gemini_key_env, "").strip()
                cache_path = out_dir / "scene_cache_gemini.json"
                try:
                    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
                except Exception:
                    cache = {}
                cache = cache if isinstance(cache, dict) else {}

                prompt_version = "v1"
                panel_crops: list[tuple[int, Image.Image, str]] = []
                for panel_idx, rect in enumerate(panel_rects):
                    crop = _crop_rect(pil_page, rect)
                    key = f"{_sha256_png(crop)}:{args.gemini_model}:{prompt_version}"
                    panel_crops.append((panel_idx, crop, key))

                by_panel: dict[int, dict] = {}
                missing: list[tuple[int, Image.Image]] = []
                missing_keys: dict[int, str] = {}

                if not api_key:
                    for panel_idx, _, key in panel_crops:
                        by_panel[panel_idx] = {"tags": [], "caption": f"ERROR: missing Gemini API key env var {args.gemini_key_env!r}", "raw": {"provider": "gemini", "cache_key": key}}
                else:
                    for panel_idx, crop, key in panel_crops:
                        cached = cache.get(key)
                        if isinstance(cached, dict) and isinstance(cached.get("caption"), str):
                            by_panel[panel_idx] = {"tags": cached.get("tags", []), "caption": cached.get("caption", "").strip(), "raw": {"provider": "gemini", "cache_key": key, "cached": True}}
                        else:
                            missing.append((panel_idx, crop))
                            missing_keys[panel_idx] = key

                    batch_size = max(1, int(args.gemini_batch_size))
                    gemini_batches_meta: list[dict] = []
                    for i in range(0, len(missing), batch_size):
                        batch = missing[i : i + batch_size]

                        def do_call():
                            return _gemini_generate_scene_json_batch(
                                api_key=api_key,
                                model=args.gemini_model,
                                panel_images=batch,
                                max_tokens=max(128, args.scene_tokens * max(1, len(batch))),
                                temperature=0.2,
                                thinking_budget=max(0, int(args.gemini_thinking_budget)),
                                timeout_s=max(5, args.gemini_timeout),
                            )

                        try:
                            items, raw = _gemini_post_with_backoff(do_call)
                            gemini_batches_meta.append(
                                {
                                    "page_idx": page_idx,
                                    "panel_indices": [pidx for pidx, _ in batch],
                                    "finishReason": raw.get("finishReason"),
                                    "usageMetadata": raw.get("usageMetadata"),
                                }
                            )
                            # If truncated, retry once with a larger output budget (still a single request).
                            if (not items) and raw.get("finishReason") == "MAX_TOKENS":
                                def do_call_retry():
                                    return _gemini_generate_scene_json_batch(
                                        api_key=api_key,
                                        model=args.gemini_model,
                                        panel_images=batch,
                                        max_tokens=min(2048, max(256, 2 * max(128, args.scene_tokens * max(1, len(batch))))),
                                        temperature=0.2,
                                        thinking_budget=max(0, int(args.gemini_thinking_budget)),
                                        timeout_s=max(5, args.gemini_timeout),
                                    )
                                items, raw = _gemini_post_with_backoff(do_call_retry)
                                gemini_batches_meta.append(
                                    {
                                        "page_idx": page_idx,
                                        "panel_indices": [pidx for pidx, _ in batch],
                                        "finishReason": raw.get("finishReason"),
                                        "usageMetadata": raw.get("usageMetadata"),
                                        "retry": True,
                                    }
                                )
                            for it in items:
                                pidx = int(it.get("panel_idx"))
                                if pidx not in missing_keys:
                                    continue
                                by_panel[pidx] = {"tags": it.get("tags", []), "caption": it.get("caption", "").strip(), "raw": {"provider": "gemini", "cache_key": missing_keys[pidx], "cached": False}}
                                cache[missing_keys[pidx]] = {"tags": it.get("tags", []), "caption": it.get("caption", "").strip()}
                            # For any still-missing entries, store the raw response for debugging once (no retries).
                            for pidx, _ in batch:
                                if pidx not in by_panel:
                                    by_panel[pidx] = {"tags": [], "caption": "ERROR: Gemini did not return an item for this panel", "raw": {"provider": "gemini", "cache_key": missing_keys.get(pidx, ""), "gemini_raw": raw}}
                        except Exception as e:
                            for pidx, _ in batch:
                                if pidx not in by_panel:
                                    by_panel[pidx] = {"tags": [], "caption": f"ERROR: {e}", "raw": {"provider": "gemini", "cache_key": missing_keys.get(pidx, ""), "cached": False}}

                    try:
                        cache_path.write_text(json.dumps(_to_jsonable(cache), ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass

                    # Write per-request token usage so you can estimate cost per panel.
                    try:
                        (out_dir / f"page_{page_idx}_gemini_usage.json").write_text(
                            json.dumps(_to_jsonable(gemini_batches_meta), ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

                for panel_idx, rect in enumerate(panel_rects):
                    info = by_panel.get(panel_idx, {"tags": [], "caption": "", "raw": {}})
                    scene_items.append(
                        {
                            "page_idx": page_idx,
                            "panel_idx": panel_idx,
                            "bbox": list(rect),
                            "tags": info.get("tags", []) if isinstance(info.get("tags"), list) else [],
                            "caption": info.get("caption", ""),
                            "raw": info.get("raw", {}),
                        }
                    )

                # AUTO fallback: fill Gemini failures with Ollama.
                if args.scene_provider == "auto":
                    caption_prompt = _ollama_caption_prompt()
                    for item in scene_items:
                        if item.get("caption", "").startswith("ERROR:") or not (item.get("caption", "") or "").strip():
                            panel_idx = int(item["panel_idx"])
                            rect = panel_rects[panel_idx]
                            crop = _crop_rect(pil_page, rect)
                            try:
                                caption_text = _ollama_generate_text(
                                    host=args.ollama_host,
                                    model=args.ollama_model,
                                    prompt=caption_prompt,
                                    image=crop,
                                    max_tokens=max(16, args.scene_tokens),
                                )
                            except Exception as e:
                                caption_text = f"ERROR: {e}"
                            try:
                                tags_text = _ollama_generate_text(
                                    host=args.ollama_host,
                                    model=args.ollama_model,
                                    prompt=_ollama_tags_prompt(caption_text),
                                    image=None,
                                    max_tokens=64,
                                    temperature=0.0,
                                )
                            except Exception as e:
                                tags_text = f"ERROR: {e}"
                            tags, _ = _parse_scene_labels(tags_text)
                            item["tags"] = tags
                            item["caption"] = caption_text.strip()
                            item["raw"] = {
                                "provider": "ollama",
                                "fallback_from": "gemini",
                                "caption_raw": caption_text,
                                "tags_raw": tags_text,
                            }

            (out_dir / f"page_{page_idx}_scene_labels.json").write_text(
                json.dumps(_to_jsonable(scene_items), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Panel captioning (+ optional character grounding)
        if args.panel_grounding:
            args.panel_captions = True

        if args.panel_captions and isinstance(page_result, dict):
            panel_boxes = page_result.get("panels", []) or []
            # Reduce to rects and drop unparseable items
            panel_rects: list[tuple[float, float, float, float]] = []
            for pb in panel_boxes:
                rect = _safe_rect_xyxy(pb)
                if rect is None:
                    continue
                panel_rects.append(rect)

            panel_rects = panel_rects[: max(0, args.max_panels)]

            panel_caption_items: list[dict] = []
            if panel_rects:
                page_img = _read_image_rgb_np(image_path)
                panel_images = [page_img for _ in panel_rects]
                input_texts = ["Caption the region in 1 short sentence: {0}"] * len(panel_rects)
                input_bboxes = [[[list(rect)]] for rect in panel_rects]
                captions = _generate_text(
                    model,
                    processor,
                    images=panel_images,
                    input_texts=input_texts,
                    input_bboxes=input_bboxes,
                    max_new_tokens=max(8, args.caption_tokens),
                    num_beams=max(1, args.beams),
                    suppress_structure_tokens=True,
                )
                for panel_idx, (rect, caption) in enumerate(zip(panel_rects, captions)):
                    panel_caption_items.append(
                        {
                            "page_idx": page_idx,
                            "panel_idx": panel_idx,
                            "bbox": list(rect),
                            "caption": caption.strip(),
                        }
                    )

                (out_dir / f"page_{page_idx}_panel_captions.json").write_text(
                    json.dumps(_to_jsonable(panel_caption_items), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                if args.panel_grounding:
                    grounded = model.predict_character_grounding(panel_images, [x["caption"] for x in panel_caption_items], processor)
                    grounded_items = []
                    for x, g in zip(panel_caption_items, grounded):
                        grounded_items.append({**x, **g})
                    (out_dir / f"page_{page_idx}_panel_grounding.json").write_text(
                        json.dumps(_to_jsonable(grounded_items), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

        # Prose generation for full page (optionally conditioned on transcript)
        if args.prose and isinstance(page_result, dict):
            # Keep prompt compact and stable.
            # Use the OCR-derived transcript lines (already filtered by essential text).
            dialogue_lines = []
            for item in transcript_items:
                t = (item.get("text") or "").strip()
                if not t:
                    continue
                speaker = item.get("speaker") or "unsure"
                dialogue_lines.append(f"<{speaker}>: {t}")
            dialogue_block = "\n".join(dialogue_lines[:30])
            prose_prompt = (
                "Write 2-4 short sentences of prose describing what happens in this manga page. "
                "Use the dialogue only as hints; do not copy it verbatim.\n"
                f"Dialogue:\n{dialogue_block}"
            )
            page_img = _read_image_rgb_np(image_path)
            prose = _generate_text(
                model,
                processor,
                images=[page_img],
                input_texts=[prose_prompt],
                input_bboxes=[[]],
                max_new_tokens=max(16, args.prose_tokens),
                num_beams=max(1, args.beams),
                suppress_structure_tokens=True,
            )[0]
            (out_dir / f"page_{page_idx}_prose.txt").write_text(prose.strip() + "\n", encoding="utf-8")

    (out_dir / "transcript.json").write_text(
        json.dumps(_to_jsonable(transcript_items_all), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "transcript.txt").write_text("\n\n".join(transcript_lines_all) + ("\n" if transcript_lines_all else ""), encoding="utf-8")

    print("Done.")
    print(f"- detections_and_associations: {type(det_assoc).__name__} (len={len(det_assoc)})")
    print(f"- ocr: {type(ocr).__name__} (len={len(ocr)})")
    print(f"- outputs: {str(out_dir)}")

    # Print a tiny summary for the first image to help confirm things look sane.
    if len(det_assoc) > 0 and isinstance(det_assoc[0], dict):
        print(f"First page keys: {sorted(det_assoc[0].keys())}")


if __name__ == "__main__":
    main()
