from __future__ import annotations

import sys


class Progress:
    def __init__(self, *, total: int, enabled: bool):
        self.total = int(total)
        self.enabled = bool(enabled)
        self._isatty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self.processed = 0
        self.generated = 0
        self.skipped = 0
        self.failed = 0
        self._last_print_processed = -1
        self._printed_once = False

    def update(
        self,
        *,
        processed_delta: int = 0,
        generated_delta: int = 0,
        skipped_delta: int = 0,
        failed_delta: int = 0,
        force: bool = False,
    ) -> None:
        self.processed += int(processed_delta)
        self.generated += int(generated_delta)
        self.skipped += int(skipped_delta)
        self.failed += int(failed_delta)

        if not self.enabled:
            return

        # Avoid spamming logs in non-tty output (but always print at least once).
        if (
            (not force)
            and self._printed_once
            and (not self._isatty)
            and (self.processed - self._last_print_processed) < 25
            and self.processed != self.total
        ):
            return

        self._last_print_processed = self.processed
        total_s = str(self.total) if self.total > 0 else "?"
        msg = (
            f"Audio: {self.processed}/{total_s} processed"
            f" | {self.generated} generated"
            f" | {self.skipped} skipped"
            f" | {self.failed} failed"
        )
        if self._isatty:
            print("\r" + msg, end="", file=sys.stderr, flush=True)
        else:
            print(msg, file=sys.stderr, flush=True)
        self._printed_once = True

    def done(self) -> None:
        if not self.enabled:
            return
        self.update(force=True)
        if self._isatty:
            print("", file=sys.stderr)

