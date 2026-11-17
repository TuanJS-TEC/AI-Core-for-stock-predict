import numpy as np
import pandas as pd

from ai_core.models.mmf import mag as M


def _preds(seed=0, n_days=30):
    """Two arms on one fold: `good` leans to the true class, `prior` is flat; a little mass sits on the empty class 2."""
    rng, rows = np.random.default_rng(seed), []
    for block in ("calib", "eval"):
        for d in pd.bdate_range("2025-01-01", periods=n_days):
            for s in "ABCD":
                y = int(rng.integers(0, 2))
                for arm, p1 in (("fusion_concat", 0.7 if y else 0.3), ("prior_fixed", 0.5), ("logit_price", 0.5), ("no_text", 0.5)):
                    rows.append(dict(fold="D", arm=arm, symbol=s, as_of_session=d, block=block, y_true=y,
                                     p_down=(1 - p1) * 0.9, p_flat=p1 * 0.9, p_up=0.1))
    return pd.DataFrame(rows)


def test_two_class_renormalises_and_r8_favours_the_informed_arm():
    ph = M.two_class(_preds())
    assert np.allclose(ph["p_down"] + ph["p_flat"], 1.0) and (ph["p_up"] == 0).all()
    a, b, c = M.r8(ph, 2)
    assert [r["so_sanh"] for r in (a, b, c)] == ["R8a", "R8b", "R8c"]
    assert a["mean"] < 0 and a["hi"] < 0 and a["verdict"] == "Có bằng chứng"
    assert np.isclose(a["mean"], -np.log(0.7) + np.log(0.5))
    s = {r["arm"]: r for r in M.arm_scores(ph, 2)}
    assert s["fusion_concat"]["auc"] == 1.0 and np.isclose(s["prior_fixed"]["accuracy"], 1 - s["prior_fixed"]["true_strong"])
