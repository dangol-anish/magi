from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from _errors import fail


@dataclass(frozen=True)
class TtsResult:
    audio: np.ndarray
    sample_rate: int


class TtsEngine:
    def synth(self, text: str) -> TtsResult:  # pragma: no cover
        raise NotImplementedError

    @property
    def meta(self) -> dict:  # pragma: no cover
        return {}


class DryRunEngine(TtsEngine):
    @property
    def meta(self) -> dict:
        return {"tts_provider": "dry_run"}


class PyKokoroEngine(TtsEngine):
    def __init__(self, *, voice: str, speed: float, cache_dir: str | None):
        from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
        from pykokoro.tokenizer import TokenizerConfig

        self._voice = voice
        self._speed = speed
        config = PipelineConfig(
            voice=voice,
            generation=GenerationConfig(speed=float(speed)),
            tokenizer_config=TokenizerConfig(spacy_model_size="sm"),
            cache_dir=cache_dir,
        )
        self._pipe = KokoroPipeline(config)

    def synth(self, text: str) -> TtsResult:
        res = self._pipe.run(text)
        audio = np.asarray(res.audio, dtype=np.float32)
        sr = int(res.sample_rate)
        return TtsResult(audio=audio, sample_rate=sr)

    @property
    def meta(self) -> dict:
        return {"tts_provider": "pykokoro", "tts_voice": self._voice, "tts_speed": self._speed}


class KokoroEngine(TtsEngine):
    def __init__(self, *, voice: str, speed: float, lang_code: str):
        from kokoro import KPipeline

        self._voice = voice
        self._speed = speed
        self._lang_code = lang_code
        self._pipe = KPipeline(lang_code=lang_code)

    def synth(self, text: str) -> TtsResult:
        # kokoro yields chunks; stitch them.
        chunks: list[np.ndarray] = []
        for _gs, _ps, audio in self._pipe(text, voice=self._voice, speed=float(self._speed)):
            if audio is None:
                continue
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not chunks:
            return TtsResult(audio=np.zeros((0,), dtype=np.float32), sample_rate=24000)
        return TtsResult(audio=np.concatenate(chunks, axis=0), sample_rate=24000)

    @property
    def meta(self) -> dict:
        return {
            "tts_provider": "kokoro",
            "tts_voice": self._voice,
            "tts_speed": self._speed,
            "tts_lang_code": self._lang_code,
        }


def pick_engine(*, voice: str, speed: float, prefer: str, lang_code: str, cache_dir: str | None) -> TtsEngine:
    prefer = (prefer or "").strip().lower()
    if prefer not in {"auto", "pykokoro", "kokoro"}:
        fail("--engine must be one of: auto, pykokoro, kokoro")

    errors: list[str] = []

    def _try_pykokoro() -> TtsEngine | None:
        try:
            return PyKokoroEngine(voice=voice, speed=speed, cache_dir=cache_dir)
        except Exception as e:
            errors.append(f"pykokoro: {e}")
            return None

    def _try_kokoro() -> TtsEngine | None:
        try:
            return KokoroEngine(voice=voice, speed=speed, lang_code=lang_code)
        except Exception as e:
            errors.append(f"kokoro: {e}")
            return None

    if prefer == "pykokoro":
        eng = _try_pykokoro()
        if eng is None:
            fail("Failed to initialize pykokoro. " + "; ".join(errors))
        return eng
    if prefer == "kokoro":
        eng = _try_kokoro()
        if eng is None:
            fail("Failed to initialize kokoro. " + "; ".join(errors))
        return eng

    eng = _try_pykokoro()
    if eng is not None:
        return eng
    eng = _try_kokoro()
    if eng is not None:
        return eng

    fail(
        "No Kokoro TTS engine available.\n"
        "- Install pykokoro: pip install pykokoro\n"
        "- Or install kokoro: pip install kokoro\n"
        f"Details: {'; '.join(errors) or '(none)'}"
    )

