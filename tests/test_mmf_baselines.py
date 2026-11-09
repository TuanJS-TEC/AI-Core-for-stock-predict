import numpy as np, pandas as pd, pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from mmf import baselines as B
from mmf import config as C
from mmf.frame import GROUPS


def toy():
    days = pd.bdate_range("2024-01-01", periods=300)
    f = pd.DataFrame({"symbol": "FPT", "as_of_session": days, "status": "ok", "label": np.resize([0, 1, 2, 2], 300)})
    f["label_end_date"] = f.as_of_session.shift(-5).fillna(days[-1] + pd.Timedelta(days=7))
    return f


def test_prior_fixed_is_laplace_from_train_only():
    f = toy()
    blocks = {"train": f.iloc[:100], "calib": f.iloc[100:200], "eval": f.iloc[200:]}
    p = B.predict_baseline("prior_fixed", blocks, f, f)
    n = np.bincount(f.label.iloc[:100], minlength=3)
    assert p["eval"][0] == pytest.approx((n + 1) / 103)


def test_prior_adaptive_uses_only_matured_labels():
    f = toy()
    blocks = {"train": f.iloc[:100], "calib": f.iloc[100:200], "eval": f.iloc[200:]}
    g = f.copy()
    t = f.as_of_session.iloc[200]
    g.loc[g.as_of_session >= t - pd.Timedelta(days=3), "label"] = 0   # label not yet matured at t
    a = B.predict_baseline("prior_adaptive", blocks, f, f)["eval"][0]
    b = B.predict_baseline("prior_adaptive", {**blocks, "eval": g.iloc[200:]}, g, g)["eval"][0]
    assert a == pytest.approx(b)
    assert a.sum() == pytest.approx(1)


def test_prior_adaptive_first_session_is_uniform():
    f = toy()
    p = B.predict_baseline("prior_adaptive", {"train": f.iloc[:100], "calib": f.iloc[:3], "eval": f.iloc[:3]}, f, f)
    assert p["eval"][0] == pytest.approx(np.full(3, 1 / 3))


def test_predict_proba_maps_missing_class():
    x = np.random.default_rng(0).normal(size=(40, 2))
    clf = HistGradientBoostingClassifier(max_iter=5).fit(x, np.resize([0, 2], 40))   # class 1 absent
    p = B._proba(clf, x)
    assert p.shape == (40, 3) and (p[:, 1] == 0).all() and p.sum(1) == pytest.approx(1)


@pytest.mark.parametrize("name", [n for n in C.BASELINES if n.split("_")[0] in ("logit", "gbm")])
def test_logit_gbm_shape_probs_and_order(name):
    f = toy()
    rng = np.random.default_rng(1)
    cols = [c for g in GROUPS.values() for c in g]
    f[cols] = rng.normal(size=(len(f), len(cols)))
    blocks = {"train": f.iloc[:150], "calib": f.iloc[150:220], "eval": f.iloc[220:]}
    p = B.predict_baseline(name, blocks, f, f)
    for k in ("calib", "eval"):
        assert p[k].shape == (len(blocks[k]), 3) and (p[k] >= 0).all() and p[k].sum(1) == pytest.approx(1)
    rev = {**blocks, "eval": blocks["eval"].iloc[::-1]}
    assert B.predict_baseline(name, rev, f, f)["eval"] == pytest.approx(p["eval"][::-1])
