"""Terminal app: track focused-at-desk streaks + classify activity via webcam."""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from datetime import datetime

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import store
from .activity import Activity, BLOCKS, COLORS, classify
from .detector import Camera
from .format import fmt_compact, fmt_duration
from .picker import choose_camera
from .thumbnail import render_frame
from .timeline import bucket_timeline, today_totals
from .vision import Vision

GRACE_SECONDS = float(os.environ.get("PRESENCE_GRACE_SECONDS", "10"))
SAMPLE_MS = int(os.environ.get("PRESENCE_SAMPLE_MS", "100"))
MIN_CONF = float(os.environ.get("PRESENCE_MIN_CONF", "0.6"))
THUMB_WIDTH = int(os.environ.get("PRESENCE_THUMB_WIDTH", "56"))
TIMELINE_MIN = int(os.environ.get("PRESENCE_TIMELINE_MIN", "15"))


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
        self.latest_keypoints: dict = {}
        self.face_detected = False
        self.pose_detected = False
        self.current_activity = Activity.AWAY
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
            vision = Vision(face_min_conf=MIN_CONF)
        except Exception as e:
            self.running = False
            print(f"detector error: {e}", file=sys.stderr)
            return

        try:
            while self.running:
                frame = cam.read()
                now = time.time()
                result = vision.analyze(frame) if frame is not None else None
                activity = classify(result) if result is not None else Activity.AWAY
                with self._lock:
                    self.latest_frame = frame
                    if result is not None:
                        self.latest_box = result.face_box
                        self.latest_keypoints = result.keypoints
                        self.face_detected = result.face_detected
                        self.pose_detected = result.pose_detected
                        is_present = result.present
                    else:
                        is_present = False
                    self.current_activity = activity
                    self.state.append_activity(now, activity.value)
                    if is_present:
                        if not self.streak_start:
                            self.streak_start = now
                        self.last_present = now
                        self.present = True
                        self.state.in_progress_started_at = self.streak_start
                        self.state.in_progress_last_seen = now
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
            vision.close()

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
                "keypoints": dict(self.latest_keypoints),
                "face_detected": self.face_detected,
                "pose_detected": self.pose_detected,
                "activity": self.current_activity,
                "activity_events": list(self.state.activity_events),
            }


# ---- rendering ----

def _streak_counter(s: dict) -> Text:
    color = "bright_green" if s["present"] else "yellow"
    line1 = Text("current streak", style="dim")
    line2 = Text(fmt_duration(s["current_ms"]), style=f"bold {color}")
    line2.append(f"   ({s['current_ms']} ms)", style="dim")
    activity: Activity = s["activity"]
    activity_color = COLORS[activity]
    badge = Text()
    badge.append("● ", style=activity_color)
    badge.append(activity.value.replace("_", " "), style=f"bold {activity_color}")
    via_bits = []
    if s["face_detected"]:
        via_bits.append("[bright_green]face[/]")
    if s["pose_detected"]:
        via_bits.append("[bright_cyan]pose[/]")
    if via_bits:
        badge.append("   via ", style="dim")
        badge.append(Text.from_markup(" + ".join(via_bits)))
    if not s["present"]:
        gone = int(s["now"] - s["last_present"])
        badge.append(f"   away {gone}s / grace {int(GRACE_SECONDS)}s", style="dim")
    return Group(line1, line2, Text(""), badge)


def _today_panel(s: dict) -> Panel:
    totals = today_totals(s["activity_events"], now=s["now"])
    present_total = sum(
        v for k, v in totals.items() if k != Activity.AWAY.value
    )
    longest_today_ms = 0
    day_start = datetime.fromtimestamp(s["now"]).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()
    for st in s["streaks"]:
        if st.ended_at >= day_start:
            longest_today_ms = max(longest_today_ms, st.duration_ms)
    if s["current_ms"] > longest_today_ms:
        longest_today_ms = s["current_ms"]

    head = Text()
    head.append("total at desk  ", style="dim")
    head.append(fmt_compact(int(present_total * 1000)), style="bold")
    head.append("       longest today  ", style="dim")
    head.append(fmt_compact(longest_today_ms), style="bold")

    breakdown = Text()
    for activity in Activity:
        secs = totals.get(activity.value, 0.0)
        if secs < 1:
            continue
        breakdown.append("  ")
        breakdown.append(BLOCKS[activity], style=COLORS[activity])
        breakdown.append(" ")
        breakdown.append(activity.value.replace("_", " "), style=COLORS[activity])
        breakdown.append(" ", style="dim")
        breakdown.append(fmt_compact(int(secs * 1000)), style="bold")
    if not breakdown.plain.strip():
        breakdown.append("  no activity yet today", style="dim")

    return Panel(
        Group(head, Text(""), breakdown),
        title=f"today · {datetime.fromtimestamp(s['now']).strftime('%a %b %d')}",
        border_style="blue",
    )


def _timeline_panel(s: dict, width: int) -> Panel:
    inner_width = max(20, width - 4)
    buckets = bucket_timeline(
        s["activity_events"],
        window_seconds=TIMELINE_MIN * 60,
        bucket_count=inner_width,
        now=s["now"],
    )
    strip = Text()
    for label in buckets:
        activity = Activity(label) if label in Activity._value2member_map_ else Activity.AWAY
        strip.append(BLOCKS[activity], style=COLORS[activity])
    legend = Text()
    for activity in Activity:
        legend.append(BLOCKS[activity], style=COLORS[activity])
        legend.append(" " + activity.value.replace("_", " ") + "   ", style="dim")
    return Panel(
        Group(strip, Text(""), legend),
        title=f"activity timeline · last {TIMELINE_MIN} min",
        border_style="blue",
    )


def _leaderboard(streaks) -> Table:
    table = Table(title="all-time longest streaks", expand=True, header_style="bold")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("duration", style="bold")
    table.add_column("started", style="dim")
    table.add_column("ended", style="dim")
    top = sorted(streaks, key=lambda x: x.duration_ms, reverse=True)[:8]
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
    return table


def render(tracker: Tracker, term_width: int = 100) -> Group:
    s = tracker.snapshot()

    streak_panel = Panel(_streak_counter(s), title="presence-streak", border_style="blue")
    if s["frame"] is not None:
        thumb = render_frame(
            s["frame"],
            width=THUMB_WIDTH,
            face_box=s["box"],
            keypoints=s["keypoints"],
        )
        thumb_panel = Panel(
            thumb,
            title="live · 10fps",
            border_style="green" if s["present"] else "yellow",
            padding=(0, 0),
        )
        header = Columns([streak_panel, thumb_panel], expand=True)
    else:
        header = streak_panel

    return Group(
        header,
        _today_panel(s),
        _timeline_panel(s, term_width),
        _leaderboard(s["streaks"]),
        Text(f"press Ctrl+C to quit  ·  grace={int(GRACE_SECONDS)}s", style="dim"),
    )


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
        with Live(
            render(tracker, console.width),
            console=console,
            refresh_per_second=10,
            screen=False,
        ) as live:
            while tracker.running:
                live.update(render(tracker, console.width))
                time.sleep(0.1)
    finally:
        tracker.running = False
        t.join(timeout=2)
        with tracker._lock:
            if tracker.streak_start:
                tracker.state.in_progress_started_at = tracker.streak_start
                tracker.state.in_progress_last_seen = tracker.last_present
                store.save(tracker.state)
        console.print("\n[dim]saved. bye.[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
