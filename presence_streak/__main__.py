"""Terminal app: track focused-at-desk streaks via webcam face detection."""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from datetime import datetime

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import store
from .detector import Camera, FaceDetector
from .format import fmt_compact, fmt_duration

GRACE_SECONDS = float(os.environ.get("PRESENCE_GRACE_SECONDS", "30"))
SAMPLE_MS = int(os.environ.get("PRESENCE_SAMPLE_MS", "500"))
MIN_CONF = float(os.environ.get("PRESENCE_MIN_CONF", "0.6"))


class Tracker:
    def __init__(self) -> None:
        self.state = store.load()
        self.streak_start = store.resume_or_new(self.state, GRACE_SECONDS)
        self.last_present = time.time()
        self.present = True
        self.running = True
        self._lock = threading.Lock()

    def end_streak(self, end_time: float) -> None:
        if end_time - self.streak_start < 1.0:
            return
        self.state.streaks.append(store.Streak(self.streak_start, end_time))
        self.state.in_progress_started_at = 0.0
        self.state.in_progress_last_seen = 0.0
        store.save(self.state)

    def detector_loop(self) -> None:
        try:
            cam = Camera()
            det = FaceDetector(min_confidence=MIN_CONF)
        except Exception as e:
            self.running = False
            print(f"detector error: {e}", file=sys.stderr)
            return

        try:
            while self.running:
                frame = cam.read()
                now = time.time()
                if frame is not None and det.has_face(frame):
                    with self._lock:
                        if not self.streak_start:
                            self.streak_start = now
                        self.last_present = now
                        self.present = True
                        self.state.in_progress_started_at = self.streak_start
                        self.state.in_progress_last_seen = now
                        store.save(self.state)
                else:
                    with self._lock:
                        self.present = False
                        if (
                            self.streak_start
                            and now - self.last_present > GRACE_SECONDS
                        ):
                            self.end_streak(self.last_present)
                            self.streak_start = 0.0
                time.sleep(SAMPLE_MS / 1000)
        finally:
            cam.close()

    def snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            current_ms = int((now - self.streak_start) * 1000) if self.streak_start else 0
            return {
                "current_ms": max(0, current_ms),
                "present": self.present,
                "last_present": self.last_present,
                "streaks": list(self.state.streaks),
                "now": now,
            }


def render(tracker: Tracker) -> Group:
    s = tracker.snapshot()
    status_color = "bright_green" if s["present"] else "yellow"
    status_text = "PRESENT" if s["present"] else "away…"

    big = Text()
    big.append("current streak  ", style="dim")
    big.append(fmt_duration(s["current_ms"]), style=f"bold {status_color}")
    big.append(f"   ({s['current_ms']} ms)", style="dim")

    sub = Text()
    sub.append("status: ", style="dim")
    sub.append(status_text, style=status_color)
    if not s["present"]:
        gone = int(s["now"] - s["last_present"])
        sub.append(f"  ·  away {gone}s / grace {int(GRACE_SECONDS)}s", style="dim")

    header = Panel(Group(big, sub), title="presence-streak", border_style="blue")

    table = Table(title="leaderboard — longest streaks", expand=True, header_style="bold")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("duration", style="bold")
    table.add_column("started", style="dim")
    table.add_column("ended", style="dim")

    top = sorted(s["streaks"], key=lambda x: x.duration_ms, reverse=True)[:10]
    if not top:
        table.add_row("-", "no completed streaks yet", "", "")
    else:
        for i, st in enumerate(top, 1):
            table.add_row(
                str(i),
                fmt_compact(st.duration_ms),
                datetime.fromtimestamp(st.started_at).strftime("%Y-%m-%d %H:%M"),
                datetime.fromtimestamp(st.ended_at).strftime("%H:%M:%S"),
            )

    recent_table = Table(title="recent streaks", expand=True, header_style="bold")
    recent_table.add_column("when", style="dim")
    recent_table.add_column("duration", style="bold")
    for st in list(reversed(s["streaks"]))[:5]:
        recent_table.add_row(
            datetime.fromtimestamp(st.started_at).strftime("%a %m-%d %H:%M"),
            fmt_compact(st.duration_ms),
        )
    if not s["streaks"]:
        recent_table.add_row("-", "-")

    footer = Text("press Ctrl+C to quit  ·  grace="+str(int(GRACE_SECONDS))+"s", style="dim")

    return Group(header, table, recent_table, footer)


def main() -> int:
    console = Console()
    tracker = Tracker()

    def handle_sigint(_sig, _frm):
        tracker.running = False

    signal.signal(signal.SIGINT, handle_sigint)

    t = threading.Thread(target=tracker.detector_loop, daemon=True)
    t.start()

    try:
        with Live(render(tracker), console=console, refresh_per_second=20, screen=False) as live:
            while tracker.running:
                live.update(render(tracker))
                time.sleep(0.05)
    finally:
        tracker.running = False
        t.join(timeout=2)
        # save in-progress so a quick restart resumes
        with tracker._lock:
            if tracker.streak_start:
                tracker.state.in_progress_started_at = tracker.streak_start
                tracker.state.in_progress_last_seen = tracker.last_present
                store.save(tracker.state)
        console.print("\n[dim]saved. bye.[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
