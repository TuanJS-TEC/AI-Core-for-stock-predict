import numpy as np
import pandas as pd

from ai_core import config as C
from ai_core.splits import folds


def test_frozen_fold_c_ends_before_test_end():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", "2026-12-31")
    d = pd.DataFrame({"date": np.repeat(dates, 2), "symbol": ["A", "B"] * len(dates)})
    for col in ("ret1", "ret5", "vol10", "y2"):
        d[col] = rng.normal(size=len(d))
    te = {name: te for name, *_, te in folds(d, 2)}["C"]
    assert te.date.max() < pd.Timestamp(C.TEST_END)
    assert C.FOLDS[-1][-1] == C.TEST_END and [f[0] for f in C.FOLDS] == ["A", "B", "C"]
