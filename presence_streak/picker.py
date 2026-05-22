"""Pick a camera at startup with arrow keys, with a live thumbnail preview."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import termios
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


def list_cameras(max_index: int = 4) -> list[tuple[int, str, np.ndarray | None]]:
    """Return [(opencv_index, human_name, sample_frame), ...] for cameras that actually open."""
    names = _macos_camera_names() if sys.platform == "darwin" else []
    found: list[tuple[int, str, np.ndarray | None]] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                name = names[len(found)] if len(found) < len(names) else f"Camera {i}"
                found.append((i, name, frame))
        cap.release()
    return found


def _frame_to_ansi(frame: np.ndarray, width: int = 40) -> str:
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


def _read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return ch + ch2 + ch3
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def pick(cameras: list[tuple[int, str, np.ndarray | None]]) -> int | None:
    if not cameras:
        return None
    if len(cameras) == 1:
        return cameras[0][0]

    sel = 0
    sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")  # hide cursor, clear screen
    sys.stdout.flush()
    try:
        while True:
            sys.stdout.write("\x1b[H\x1b[J")  # move home + clear to end
            sys.stdout.write(
                "pick a camera (↑/↓, enter to confirm, q to quit)\r\n"
                "the highlighted camera's live frame is shown below\r\n\r\n"
            )
            for i, (idx, name, _) in enumerate(cameras):
                marker = "▶" if i == sel else " "
                style = "\x1b[1;32m" if i == sel else "\x1b[0m"
                sys.stdout.write(f"  {style}{marker} [{idx}] {name}\x1b[0m\r\n")
            sys.stdout.write("\r\n")
            frame = cameras[sel][2]
            if frame is not None:
                thumb = _frame_to_ansi(frame, width=48)
                for line in thumb.split("\n"):
                    sys.stdout.write(line + "\r\n")
            sys.stdout.flush()

            key = _read_key()
            if key in ("\x1b[A", "k"):
                sel = (sel - 1) % len(cameras)
                # refresh frame for the newly-selected camera
                cameras[sel] = _refresh_frame(cameras[sel])
            elif key in ("\x1b[B", "j"):
                sel = (sel + 1) % len(cameras)
                cameras[sel] = _refresh_frame(cameras[sel])
            elif key in ("\r", "\n"):
                sys.stdout.write("\x1b[2J\x1b[H")
                sys.stdout.flush()
                return cameras[sel][0]
            elif key in ("q", "\x03") or key.startswith("\x1b"):
                if key == "\x1b" or key == "q" or key == "\x03":
                    return None
    finally:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()


def _refresh_frame(
    entry: tuple[int, str, np.ndarray | None],
) -> tuple[int, str, np.ndarray | None]:
    idx, name, _ = entry
    cap = cv2.VideoCapture(idx)
    frame = None
    if cap.isOpened():
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return (idx, name, frame)


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
