"""Duration formatting helpers."""
from __future__ import annotations


def fmt_duration(ms: int) -> str:
    total_s, msec = divmod(ms, 1000)
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s {msec:03d}ms"


def fmt_compact(ms: int) -> str:
    total_s = ms // 1000
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
