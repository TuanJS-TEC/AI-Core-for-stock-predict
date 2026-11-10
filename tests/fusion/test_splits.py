import subprocess
import sys

import numpy as np
import pandas as pd

from ai_core.fusion import config as C
from ai_core.fusion.splits import FOLDS, folds


def test_frozen_fold_c_ends_before_test_end():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", "2026-12-31")
    d = pd.DataFrame({"date": np.repeat(dates, 2), "symbol": ["A", "B"] * len(dates)})
    for col in ("ret1", "ret5", "vol10", "y2"):
        d[col] = rng.normal(size=len(d))
    te = {name: te for name, *_, te in folds(d, 2)}["C"]
    assert te.date.max() < pd.Timestamp(C.TEST_END)
    assert FOLDS[-1][-1] == C.TEST_END and [f[0] for f in FOLDS] == ["A", "B", "C"]


def test_config_and_finhay_do_not_import_nlp_module():
    """The 15:40 finhay cron is HTTP-only: it must not break on an ai_core.news import error."""
    code = ("import sys, ai_core.fusion.config, ai_core.fusion.finhay; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('ai_core.news', 'sklearn')); "
            "assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=C.ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
