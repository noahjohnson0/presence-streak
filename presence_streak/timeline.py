"""Aggregations on top of the activity event log."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Iterable

from .activity import Activity
from .store import ActivityEvent


def _iter_intervals(
    events: list[ActivityEvent], end_ts: float
) -> Iterable[tuple[float, float, str]]:
    """Yield (start, end, activity) intervals from a list of events. The last
    event is closed at end_ts."""
    for i, ev in enumerate(events):
        start = ev.ts
        stop = events[i + 1].ts if i + 1 < len(events) else end_ts
        if stop > start:
            yield start, stop, ev.activity


def today_totals(events: list[ActivityEvent], now: float | None = None) -> dict[str, float]:
    """Total seconds spent in each activity since midnight local."""
    if now is None:
        now = time.time()
    day_start = datetime.fromtimestamp(now).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()
    totals: dict[str, float] = {a.value: 0.0 for a in Activity}
    for start, stop, activity in _iter_intervals(events, now):
        if stop <= day_start:
            continue
        clipped_start = max(start, day_start)
        clipped_stop = min(stop, now)
        if clipped_stop > clipped_start:
            totals[activity] = totals.get(activity, 0.0) + (clipped_stop - clipped_start)
    return totals


def bucket_timeline(
    events: list[ActivityEvent],
    window_seconds: float,
    bucket_count: int,
    now: float | None = None,
) -> list[str]:
    """Return `bucket_count` activity labels (oldest first), one per equal-sized
    time bucket spanning the last `window_seconds`. The dominant activity in
    each bucket wins."""
    if now is None:
        now = time.time()
    if bucket_count <= 0 or window_seconds <= 0:
        return []
    window_start = now - window_seconds
    bucket_size = window_seconds / bucket_count
    sums: list[dict[str, float]] = [
        {a.value: 0.0 for a in Activity} for _ in range(bucket_count)
    ]
    for start, stop, activity in _iter_intervals(events, now):
        if stop <= window_start or start >= now:
            continue
        s = max(start, window_start)
        e = min(stop, now)
        # Distribute the interval across the buckets it overlaps.
        b_start = int((s - window_start) / bucket_size)
        b_end = int((e - window_start) / bucket_size)
        for b in range(max(0, b_start), min(bucket_count - 1, b_end) + 1):
            bs = window_start + b * bucket_size
            be = bs + bucket_size
            overlap = min(e, be) - max(s, bs)
            if overlap > 0:
                sums[b][activity] = sums[b].get(activity, 0.0) + overlap
    labels: list[str] = []
    for bucket in sums:
        if not any(v > 0 for v in bucket.values()):
            labels.append(Activity.AWAY.value)
            continue
        labels.append(max(bucket.items(), key=lambda kv: kv[1])[0])
    return labels
