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
from .picker import choose_camera
from .thumbnail import render_frame

GRACE_SECONDS = float(os.environ.get("PRESENCE_GRACE_SECONDS", "10"))
SAMPLE_MS = int(os.environ.get("PRESENCE_SAMPLE_MS", "100"))  # 10fps capture+detect
MIN_CONF = float(os.environ.get("PRESENCE_MIN_CONF", "0.6"))
THUMB_WIDTH = int(os.environ.get("PRESENCE_THUMB_WIDTH", "56"))


class Tracker:
    def __init__(self, camera_index: int) -> None:
        self.camera_index = camera_index
        self.state = store.load()
        self.streak_start = store.resume_or_new(self.state, GRACE_SECONDS)
        self.last_present = time.time()
        self.present = True
        self.running = True
        self._lock = threading.Lock()
        self.latest_frame = None
        self.latest_box: tuple[int, int, int, int] | None = None
        self.last_save = 0.0

    def end_streak(self, end_time: float) -> None:
        if end_time - self.streak_start < 1.0:
            return
        self.state.streaks.append(store.Streak(self.streak_start, end_time))
        self.state.in_progress_started_at = 0.0
        self.state.in_progress_last_seen = 0.0
        store.save(self.state)

    def detector_loop(self) -> None:
        try:
            cam = Camera(self.camera_index)
            det = FaceDetector(min_confidence=MIN_CONF)
        except Exception as e:
            self.running = False
            print(f"detector error: {e}", file=sys.stderr)
            return

        try:
            while self.running:
                frame = cam.read()
                now = time.time()
                has_face, box = (False, None)
                if frame is not None:
                    has_face, box = det.detect(frame)
                with self._lock:
                    self.latest_frame = frame
                    self.latest_box = box
                    if has_face:
                        if not self.streak_start:
                            self.streak_start = now
                        self.last_present = now
                        self.present = True
                        self.state.in_progress_started_at = self.streak_start
                        self.state.in_progress_last_seen = now
                        # throttle disk writes to ~1Hz; the live UI doesn't need them faster
                        if now - self.last_save > 1.0:
                            store.save(self.state)
                            self.last_save = now
                    else:
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
                "frame": self.latest_frame,
                "box": self.latest_box,
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

    if s["frame"] is not None:
        thumb = render_frame(s["frame"], width=THUMB_WIDTH, face_box=s["box"])
        thumb_panel = Panel(thumb, title="webcam (10fps)", border_style="green" if s["present"] else "yellow", padding=(0, 0))
        from rich.columns import Columns
        header = Columns([
            Panel(Group(big, sub), title="presence-streak", border_style="blue"),
            thumb_panel,
        ], expand=True)
    else:
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
    args = sys.argv[1:]
    keep = "--keep" in args or "-k" in args
    explicit: int | None = None
    for a in args:
        if a.startswith("--camera="):
            explicit = int(a.split("=", 1)[1])
    if explicit is not None:
        cam_index = explicit
    else:
        cam_index = choose_camera(force=not keep)
    if cam_index is None:
        console.print("[red]no camera selected[/red]")
        return 1
    tracker = Tracker(cam_index)

    def handle_sigint(_sig, _frm):
        tracker.running = False

    signal.signal(signal.SIGINT, handle_sigint)

    t = threading.Thread(target=tracker.detector_loop, daemon=True)
    t.start()

    try:
        with Live(render(tracker), console=console, refresh_per_second=10, screen=False) as live:
            while tracker.running:
                live.update(render(tracker))
                time.sleep(0.1)
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
