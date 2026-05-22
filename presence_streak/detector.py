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

    def has_face(self, frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        for i in range(detections.shape[2]):
            if float(detections[0, 0, i, 2]) >= self.min_confidence:
                return True
        return False


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
