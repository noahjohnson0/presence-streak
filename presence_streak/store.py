"""Persist completed streaks + in-progress streak."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_PATH = Path.home() / ".presence_streak" / "state.json"


@dataclass
class Streak:
    started_at: float
    ended_at: float

    @property
    def duration_ms(self) -> int:
        return int((self.ended_at - self.started_at) * 1000)


@dataclass
class ActivityEvent:
    """Marks the start of a period spent in a given activity class. The period
    runs until the next event (or until in_progress_last_seen, for the open
    tail)."""
    ts: float
    activity: str


@dataclass
class State:
    streaks: list[Streak] = field(default_factory=list)
    # if non-zero, a streak was in progress when the app last exited
    in_progress_started_at: float = 0.0
    in_progress_last_seen: float = 0.0
    activity_events: list[ActivityEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "streaks": [asdict(s) for s in self.streaks],
            "in_progress_started_at": self.in_progress_started_at,
            "in_progress_last_seen": self.in_progress_last_seen,
            "activity_events": [asdict(e) for e in self.activity_events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        return cls(
            streaks=[Streak(**s) for s in d.get("streaks", [])],
            in_progress_started_at=d.get("in_progress_started_at", 0.0),
            in_progress_last_seen=d.get("in_progress_last_seen", 0.0),
            activity_events=[ActivityEvent(**e) for e in d.get("activity_events", [])],
        )

    def append_activity(self, ts: float, activity: str) -> None:
        """Append only if the activity is different from the most recent event,
        which collapses runs of the same class into a single time interval."""
        if self.activity_events and self.activity_events[-1].activity == activity:
            return
        self.activity_events.append(ActivityEvent(ts=ts, activity=activity))


def load() -> State:
    if not STATE_PATH.exists():
        return State()
    try:
        return State.from_dict(json.loads(STATE_PATH.read_text()))
    except Exception:
        return State()


def save(state: State) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2))
    tmp.replace(STATE_PATH)


def resume_or_new(state: State, grace_seconds: float) -> float:
    """Return the start time of the active streak — resume the prior one if
    we exited recently, otherwise start fresh at now()."""
    now = time.time()
    if (
        state.in_progress_started_at
        and now - state.in_progress_last_seen <= grace_seconds
    ):
        return state.in_progress_started_at
    # finalize stale in-progress streak if any
    if state.in_progress_started_at and state.in_progress_last_seen > state.in_progress_started_at:
        state.streaks.append(Streak(state.in_progress_started_at, state.in_progress_last_seen))
    return now
