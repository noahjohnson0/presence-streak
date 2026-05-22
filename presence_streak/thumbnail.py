"""Render a webcam frame as a truecolor terminal thumbnail using half-blocks."""
from __future__ import annotations

import cv2
import numpy as np
from rich.text import Text


def render_frame(
    frame: np.ndarray,
    width: int = 56,
    face_box: tuple[int, int, int, int] | None = None,
) -> Text:
    h, w = frame.shape[:2]
    new_w = width
    new_h = max(2, int(h * width / w / 2) * 2)
    small = cv2.resize(frame, (new_w, new_h))

    if face_box is not None:
        x1, y1, x2, y2 = face_box
        sx = new_w / w
        sy = new_h / h
        bx1, by1 = max(0, int(x1 * sx)), max(0, int(y1 * sy))
        bx2, by2 = min(new_w - 1, int(x2 * sx)), min(new_h - 1, int(y2 * sy))
        cv2.rectangle(small, (bx1, by1), (bx2, by2), (0, 255, 0), 1)

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    text = Text()
    for y in range(0, new_h, 2):
        for x in range(new_w):
            t = rgb[y, x]
            b = rgb[y + 1, x] if y + 1 < new_h else t
            text.append(
                "▀",
                style=f"rgb({int(t[0])},{int(t[1])},{int(t[2])}) on rgb({int(b[0])},{int(b[1])},{int(b[2])})",
            )
        text.append("\n")
    return text
