from __future__ import annotations

import time

from _tts import TtsEngine, TtsResult


def synth_with_retry(
    *,
    engine: TtsEngine,
    text: str,
    attempts: int,
    base_wait_s: float,
    backoff: float,
) -> TtsResult:
    if attempts <= 0:
        attempts = 1
    wait_s = max(0.0, float(base_wait_s))
    for attempt_idx in range(1, attempts + 1):
        try:
            return engine.synth(text)
        except KeyboardInterrupt:
            raise
        except Exception:
            if attempt_idx >= attempts:
                raise
            if wait_s > 0:
                time.sleep(wait_s)
            wait_s *= float(backoff) if float(backoff) > 0 else 1.0

