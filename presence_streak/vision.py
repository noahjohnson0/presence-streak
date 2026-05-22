"""Combined face + pose detector.

Why combined: face-only detection drops the streak whenever the user
leans forward with a hat brim, looks down at the keyboard, or covers
their face with a hand. MediaPipe Pose still sees the shoulders and
torso in all of those cases, so we OR the two signals together for
the "are they at the desk" decision. The pose keypoints are also
the raw input for the activity classifier downstream.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .detector import FaceDetector

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = Path.home() / ".presence_streak" / "models" / "pose_landmarker_lite.task"


def _download_pose_model() -> Path:
    POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not POSE_MODEL_PATH.exists() or POSE_MODEL_PATH.stat().st_size == 0:
        with urllib.request.urlopen(POSE_MODEL_URL) as r, open(POSE_MODEL_PATH, "wb") as f:
            f.write(r.read())
    return POSE_MODEL_PATH

# Subset of MediaPipe Pose landmark indices we care about (the upper body).
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

UPPER_BODY_LANDMARKS = [
    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
]


@dataclass
class Keypoint:
    x: int  # pixel coords in original frame
    y: int
    visibility: float


@dataclass
class VisionResult:
    face_box: tuple[int, int, int, int] | None
    face_detected: bool
    pose_detected: bool
    # Map of landmark-index -> Keypoint. Only includes ones with visibility>0.
    keypoints: dict[int, Keypoint]

    @property
    def present(self) -> bool:
        """At-the-desk decision. Generous: any of these is enough."""
        if self.face_detected:
            return True
        if not self.pose_detected:
            return False
        # Either shoulder visible is enough — the user is in frame even if
        # their face is occluded by a hat brim, the keyboard, etc.
        for idx in (LEFT_SHOULDER, RIGHT_SHOULDER, NOSE, LEFT_EAR, RIGHT_EAR):
            kp = self.keypoints.get(idx)
            if kp is not None and kp.visibility >= 0.5:
                return True
        return False


class Vision:
    """One-stop detector: runs face + pose on each frame."""

    def __init__(self, face_min_conf: float = 0.6) -> None:
        self.face = FaceDetector(min_confidence=face_min_conf)
        model_path = _download_pose_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp_vision.PoseLandmarker.create_from_options(options)

    def analyze(self, frame: np.ndarray) -> VisionResult:
        face_ok, face_box = self.face.detect(frame)

        h, w = frame.shape[:2]
        target_w = 320
        scale = target_w / w
        small = cv2.resize(frame, (target_w, int(h * scale)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.pose.detect(mp_image)

        keypoints: dict[int, Keypoint] = {}
        pose_ok = False
        if result.pose_landmarks:
            pose_ok = True
            landmarks = result.pose_landmarks[0]
            for idx in UPPER_BODY_LANDMARKS:
                lm = landmarks[idx]
                keypoints[idx] = Keypoint(
                    x=int(lm.x * w),
                    y=int(lm.y * h),
                    visibility=float(lm.visibility),
                )

        return VisionResult(
            face_box=face_box,
            face_detected=face_ok,
            pose_detected=pose_ok,
            keypoints=keypoints,
        )

    def close(self) -> None:
        self.pose.close()
