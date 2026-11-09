import numpy as np, pytest
from mmf import evaluate as E

P = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
Y = np.array([0, 1, 2])

def test_logloss_and_brier_by_hand():
    assert E.row_logloss(Y, P) == pytest.approx(-np.log(.8) * np.ones(3))
    assert E.row_brier(Y, P) == pytest.approx(np.ones(3) * (.04 + .01 + .01))

def test_perfect_scores():
    s = E.scores(Y, P)
    assert s["macro_f1"] == 1 and s["accuracy"] == 1 and s["auc_up_down"] == 1 and s["n"] == 3

def test_temperature_recovers_scale_and_keeps_argmax():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 4000)
    z = np.eye(3)[y] * 1.0 + rng.normal(0, 1, (4000, 3))
    t = E.fit_temperature(z * 5, y)
    assert t > 2
    assert (E.apply_temperature(z, t).argmax(1) == z.argmax(1)).all()

def test_alpha_shifts_toward_rare_class():
    p = np.tile([.45, .4, .15], (9, 1)); y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert (E.decide(p, 1.0, np.array([.45, .45, .1])) == 2).all()
    assert (E.decide(p, 0.0, np.array([.45, .45, .1])) == 0).all()

def test_paired_diff_sign_and_ci():
    d = np.repeat(np.arange(200), 5)
    m, lo, hi = E.paired_diff(np.full(1000, .9), np.full(1000, 1.0), d)
    assert m == pytest.approx(-.1) and hi < 0
    rng = np.random.default_rng(1)
    m, lo, hi = E.paired_diff(rng.normal(1, .3, 1000), rng.normal(1, .3, 1000), d)
    assert lo < 0 < hi

def test_verdict_three_outcomes():
    assert E.verdict(-.01, -.001, 2, 3) == "Có bằng chứng"
    assert E.verdict(-.01, .001, 3, 3) == "Có tín hiệu, bằng chứng yếu"
    assert E.verdict(-.01, -.001, 1, 3) == "Có tín hiệu, bằng chứng yếu"
    assert E.verdict(.01, .02, 0, 3) == "Chưa chứng minh được"
