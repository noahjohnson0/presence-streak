"""Pick a camera at startup with arrow keys, with a streaming live preview."""
from __future__ import annotations

import json
import re
import select
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path.home() / ".presence_streak" / "config.json"


def _macos_camera_names() -> list[str]:
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"], text=True, timeout=4
        )
    except Exception:
        return []
    names: list[str] = []
    for line in out.splitlines():
        m = re.match(r"^    ([^ ].+?):\s*$", line)
        if m and m.group(1) != "Camera":
            names.append(m.group(1))
    return names


def list_cameras(max_index: int = 4) -> list[tuple[int, str]]:
    """Return [(opencv_index, human_name), ...] for cameras that actually open."""
    names = _macos_camera_names() if sys.platform == "darwin" else []
    found: list[tuple[int, str]] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                name = names[len(found)] if len(found) < len(names) else f"Camera {i}"
                found.append((i, name))
        cap.release()
    return found


def _frame_to_ansi(frame: np.ndarray, width: int = 48) -> str:
    h, w = frame.shape[:2]
    nw = width
    nh = max(2, int(h * width / w / 2) * 2)
    small = cv2.resize(frame, (nw, nh))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    lines = []
    for y in range(0, nh, 2):
        parts = []
        for x in range(nw):
            t = rgb[y, x]
            b = rgb[y + 1, x] if y + 1 < nh else t
            parts.append(
                f"\x1b[38;2;{int(t[0])};{int(t[1])};{int(t[2])};"
                f"48;2;{int(b[0])};{int(b[1])};{int(b[2])}m▀"
            )
        parts.append("\x1b[0m")
        lines.append("".join(parts))
    return "\n".join(lines)


def _read_available_key() -> str | None:
    """Non-blocking single keystroke read (must be in raw mode already)."""
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if not r:
        return None
    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch
    # could be a bare ESC or a CSI sequence — peek with a short timeout
    r2, _, _ = select.select([sys.stdin], [], [], 0.02)
    if not r2:
        return ch
    ch2 = sys.stdin.read(1)
    if ch2 != "[":
        return ch + ch2
    r3, _, _ = select.select([sys.stdin], [], [], 0.02)
    if not r3:
        return ch + ch2
    ch3 = sys.stdin.read(1)
    return ch + ch2 + ch3


def pick(cameras: list[tuple[int, str]]) -> int | None:
    if not cameras:
        return None
    if len(cameras) == 1:
        return cameras[0][0]

    sel = 0
    cap: cv2.VideoCapture | None = None
    current_open = -1
    last_frame: np.ndarray | None = None
    warmup_until = 0.0

    def open_sel() -> None:
        nonlocal cap, current_open, warmup_until, last_frame
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(cameras[sel][0])
        current_open = sel
        warmup_until = time.time() + 0.6  # let auto-exposure settle
        last_frame = None

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")
    sys.stdout.flush()
    try:
        tty.setraw(fd)
        open_sel()
        while True:
            if cap is not None:
                ok, frame = cap.read()
                if ok and frame is not None:
                    last_frame = frame

            # render
            sys.stdout.write("\x1b[H\x1b[J")
            sys.stdout.write(
                "pick a camera (↑/↓, enter to confirm, q to quit)\r\n"
                "live preview below — the highlighted camera is streaming\r\n\r\n"
            )
            for i, (idx, name) in enumerate(cameras):
                marker = "▶" if i == sel else " "
                style = "\x1b[1;32m" if i == sel else "\x1b[0m"
                sys.stdout.write(f"  {style}{marker} [{idx}] {name}\x1b[0m\r\n")
            sys.stdout.write("\r\n")
            if last_frame is not None and time.time() >= warmup_until:
                for line in _frame_to_ansi(last_frame, width=48).split("\n"):
                    sys.stdout.write(line + "\r\n")
            else:
                sys.stdout.write("\x1b[2m  warming up camera…\x1b[0m\r\n")
            sys.stdout.flush()

            key = _read_available_key()
            if key is None:
                time.sleep(0.05)
                continue
            if key in ("\x1b[A", "k"):
                sel = (sel - 1) % len(cameras)
                open_sel()
            elif key in ("\x1b[B", "j"):
                sel = (sel + 1) % len(cameras)
                open_sel()
            elif key in ("\r", "\n"):
                return cameras[sel][0]
            elif key in ("q", "\x03", "\x1b"):
                return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if cap is not None:
            cap.release()
        sys.stdout.write("\x1b[?25h\x1b[2J\x1b[H")
        sys.stdout.flush()


def load_saved_index() -> int | None:
    if not CONFIG_PATH.exists():
        return None
    try:
        return int(json.loads(CONFIG_PATH.read_text()).get("camera_index"))
    except Exception:
        return None


def save_index(index: int) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({"camera_index": index}))


def choose_camera(force: bool = False) -> int | None:
    if not force:
        saved = load_saved_index()
        if saved is not None:
            return saved
    cams = list_cameras()
    if not cams:
        print("no cameras available", file=sys.stderr)
        return None
    chosen = pick(cams)
    if chosen is not None:
        save_index(chosen)
    return chosen
