"""Pick a camera at startup with arrow keys, and remember the choice."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import termios
import tty
from pathlib import Path

import cv2

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


def pick(cameras: list[tuple[int, str]]) -> int | None:
    if not cameras:
        return None
    if len(cameras) == 1:
        return cameras[0][0]

    sel = 0
    print("\x1b[?25l", end="")  # hide cursor
    try:
        first = True
        while True:
            if not first:
                # move cursor up to overwrite previous render
                print(f"\x1b[{len(cameras) + 2}A", end="")
            first = False
            print("pick a camera (↑/↓, enter):                  ")
            for i, (idx, name) in enumerate(cameras):
                marker = "▶" if i == sel else " "
                style_on = "\x1b[1;32m" if i == sel else "\x1b[2m"
                print(f"  {style_on}{marker} [{idx}] {name}\x1b[0m" + " " * 20)
            print("\x1b[2m  esc to quit\x1b[0m" + " " * 20)
            key = _read_key()
            if key in ("\x1b[A", "k"):
                sel = (sel - 1) % len(cameras)
            elif key in ("\x1b[B", "j"):
                sel = (sel + 1) % len(cameras)
            elif key in ("\r", "\n", " "):
                return cameras[sel][0]
            elif key in ("\x1b", "q", "\x03"):
                return None
    finally:
        print("\x1b[?25h", end="")  # show cursor


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
    """Return the chosen OpenCV camera index, prompting if needed."""
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
