"""Rule-based activity classification from face + pose keypoints.

Inputs come from vision.VisionResult — face_box and an upper-body keypoint
dict. The classifier is intentionally simple and stateless; the timeline
logic in store.py dedupes consecutive same-class samples into events.
"""
from __future__ import annotations

from enum import Enum

from .vision import (
    LEFT_EAR, LEFT_EYE, LEFT_SHOULDER,
    NOSE,
    RIGHT_EAR, RIGHT_EYE, RIGHT_SHOULDER,
    VisionResult,
)


class Activity(str, Enum):
    AWAY = "away"
    FOCUSED = "focused"          # face visible, head centered + upright
    LEANING = "leaning"          # body present, head somewhat lowered
    HEAD_DOWN = "head_down"      # nose near / below shoulder line (typing, reading desk)
    LOOKING_AWAY = "looking_away"  # face yaw — clearly turned to one side


# Display colors per activity, used by the UI.
COLORS: dict[Activity, str] = {
    Activity.AWAY:         "grey50",
    Activity.FOCUSED:      "bright_green",
    Activity.LEANING:      "yellow",
    Activity.HEAD_DOWN:    "cyan",
    Activity.LOOKING_AWAY: "magenta",
}

# Single-char block used by the timeline strip.
BLOCKS: dict[Activity, str] = {
    Activity.AWAY:         "·",
    Activity.FOCUSED:      "█",
    Activity.LEANING:      "▓",
    Activity.HEAD_DOWN:    "▒",
    Activity.LOOKING_AWAY: "░",
}


def _kp(result: VisionResult, idx: int, min_vis: float = 0.5):
    kp = result.keypoints.get(idx)
    if kp is None or kp.visibility < min_vis:
        return None
    return kp


def classify(result: VisionResult) -> Activity:
    if not result.present:
        return Activity.AWAY

    ls = _kp(result, LEFT_SHOULDER)
    rs = _kp(result, RIGHT_SHOULDER)
    nose = _kp(result, NOSE, min_vis=0.3)

    # If we don't have both shoulders we can't reason about posture geometry.
    # Treat face-only detection as FOCUSED — the user is plainly visible.
    if ls is None or rs is None:
        return Activity.FOCUSED if result.face_detected else Activity.LEANING

    # Asymmetric ear visibility = head turned sideways. Strong signal.
    le, re = _kp(result, LEFT_EAR, 0.0), _kp(result, RIGHT_EAR, 0.0)
    if le is not None and re is not None:
        if abs(le.visibility - re.visibility) > 0.45:
            return Activity.LOOKING_AWAY

    # Same idea via eyes — more reliable when wearing a hat that hides ears.
    le_eye, re_eye = _kp(result, LEFT_EYE, 0.0), _kp(result, RIGHT_EYE, 0.0)
    if le_eye is not None and re_eye is not None:
        if abs(le_eye.visibility - re_eye.visibility) > 0.45:
            return Activity.LOOKING_AWAY

    shoulder_width = abs(ls.x - rs.x)
    shoulder_y = (ls.y + rs.y) / 2
    if shoulder_width < 30 or nose is None:
        # Pose was found but we can't measure head position reliably — call it
        # head_down rather than dropping presence entirely.
        return Activity.HEAD_DOWN

    # Image coords: y increases downward, so "nose above shoulders" = nose.y < shoulder.y.
    # neck_extension > 0 when the head is held up.
    neck_extension = shoulder_y - nose.y
    ratio = neck_extension / shoulder_width

    if ratio < 0.35:
        return Activity.HEAD_DOWN
    if ratio < 0.75:
        return Activity.LEANING
    return Activity.FOCUSED
