from __future__ import annotations


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def short_exc(e: BaseException, *, max_len: int = 280) -> str:
    msg = str(e).strip().replace("\n", " ")
    if len(msg) > max_len:
        msg = msg[: max_len - 3] + "..."
    return msg or e.__class__.__name__

