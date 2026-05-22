"""Face presence detector using OpenCV's ResNet-SSD DNN model."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path.home() / ".presence_streak" / "models"
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
WEIGHTS_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())


class FaceDetector:
    def __init__(self, min_confidence: float = 0.6) -> None:
        proto = MODEL_DIR / "deploy.prototxt"
        weights = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
        _download(PROTO_URL, proto)
        _download(WEIGHTS_URL, weights)
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        self.min_confidence = min_confidence

    def detect(self, frame: np.ndarray) -> tuple[bool, tuple[int, int, int, int] | None]:
        """Return (has_face, best_box_xyxy_in_frame_coords)."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        best_conf = 0.0
        best_box: tuple[int, int, int, int] | None = None
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf >= self.min_confidence and conf > best_conf:
                best_conf = conf
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                best_box = (x1, y1, x2, y2)
        return (best_box is not None, best_box)

    def has_face(self, frame: np.ndarray) -> bool:
        return self.detect(frame)[0]


class Camera:
    def __init__(self, index: int = 0) -> None:
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam — grant camera access and retry.")

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()
