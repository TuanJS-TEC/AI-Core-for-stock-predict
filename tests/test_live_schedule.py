"""jobs/live_update.sh cadence, checked with LIVE_DRY=1 (steps are printed, never run) and a fake LIVE_CLOCK."""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "jobs" / "live_update.sh"
NEWS = ["collect", "process", "news-push"]


@pytest.mark.parametrize("clock,steps", [
    ("2026-11-12 03:04", []),                                    # night, not a 15-min mark
    ("2026-11-12 03:00", NEWS),                                  # night, 15-min mark
    ("2026-11-12 10:02", NEWS),                                  # day, between marks
    ("2026-11-12 10:00", NEWS + ["quotes"]),
    ("2026-11-12 15:16", NEWS + ["predict", "rangecast", "live", "forecast-push"]),
    ("2026-11-12 15:20", NEWS),
    ("2026-11-14 10:00", NEWS),                                  # Saturday: no quotes
    ("2026-11-12 22:00", NEWS),                                  # 22:00 is still a 15-min mark
    ("2026-11-12 22:02", []),
])
def test_schedule(clock, steps):
    env = {**os.environ, "LIVE_DRY": "1", "LIVE_CLOCK": clock}
    out = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == steps
