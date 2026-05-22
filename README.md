# presence-streak

Terminal app that tracks how long you spend in front of your computer in unbroken streaks.

Uses your webcam + an on-device ML face detector (OpenCV DNN, ResNet-SSD) to decide whether you're present. As long as a face is detected, the streak keeps counting up — shown live in `ms / sec / min / hr`. Step away long enough and the streak ends and goes on the leaderboard.

## Install

```bash
cd ~/repos/presence-streak
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run will prompt macOS for camera access.

## Run

```bash
python -m presence_streak
```

Every launch shows an arrow-key picker listing every webcam attached (named via `system_profiler` on macOS). Pass `--keep` (or `-k`) to skip the prompt and reuse your last choice, or `--camera=N` to pick by index. Saved to `~/.presence_streak/config.json`.

Press `Ctrl+C` to quit. Your current streak is saved on exit so a quick quit doesn't kill it.

## How it works

- Samples a webcam frame every ~500ms
- Runs OpenCV's ResNet-SSD face detector (auto-downloaded on first launch)
- Streak stays alive across gaps shorter than `GRACE_SECONDS` (default 30s) so blinks / leaning out of frame don't reset you
- Streaks persist to `~/.presence_streak/state.json`

## Config

Env vars:

- `PRESENCE_GRACE_SECONDS` — how long you can be missing before the streak breaks (default 10)
- `PRESENCE_SAMPLE_MS` — sample interval (default 500)
- `PRESENCE_MIN_CONF` — face detector confidence threshold (default 0.6)
- `PRESENCE_THUMB_WIDTH` — terminal thumbnail width in chars (default 56)
- `PRESENCE_SAMPLE_MS` — capture/detect interval (default 100 = ~10fps)
