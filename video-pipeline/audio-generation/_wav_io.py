from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from _errors import fail


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_f = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio_f = np.clip(audio_f, -1.0, 1.0)
    pcm = (audio_f * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        n_channels = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        n_frames = int(wf.getnframes())
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        fail(f"Unsupported WAV sample width (expected int16): {path}")

    audio_i16 = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        audio_i16 = audio_i16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    audio_f = (audio_i16.astype(np.float32)) / 32768.0
    return audio_f, sr


def silence(sr: int, ms: float) -> np.ndarray:
    n = int(round(sr * (ms / 1000.0)))
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    return np.zeros((n,), dtype=np.float32)


def samples_to_ms(samples: int, sr: int) -> int:
    if sr <= 0:
        return 0
    return int(round((samples / float(sr)) * 1000.0))

