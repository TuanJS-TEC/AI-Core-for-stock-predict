import numpy as np, torch
from ai_core.models.mmf import config as C, train as TR
from ai_core.models.mmf.models import FusionNet
from ai_core.models.mmf.windows import Samples

def toy(n, seed, signal=True):
    r = np.random.default_rng(seed)
    y = r.integers(0, 3, n)
    p = r.normal(0, 1, (n, C.LP, 25)).astype("float32")
    if signal:
        p[:, -1, 0] += (y - 1) * 3
    return Samples(p, np.zeros((n, C.LT, 14), "float32"), np.zeros((n, 10), "float32"), np.zeros(n, "int64"),
                   y.astype("int64"), np.array(["FPT"] * n), np.arange(n), np.arange(n))

def net():
    return FusionNet(("price",), "none", 15, 25, 14, 10, "gru")

def test_learns_planted_signal_better_than_prior():
    tr, tu = toy(600, 0), toy(200, 1)
    out = TR.fit_net(net, tr, tu, 59, "cpu", max_epochs=40)
    assert out["tune_nll"] < 0.8 * np.log(3)

def test_same_seed_reproduces_on_cpu():
    tr, tu = toy(200, 0), toy(50, 1)
    x = TR.fit_net(net, tr, tu, 59, "cpu", max_epochs=3)["tune_nll"]
    y = TR.fit_net(net, tr, tu, 59, "cpu", max_epochs=3)["tune_nll"]
    assert x == y

def test_returns_best_epoch_not_last():
    tr, tu = toy(60, 0, signal=False), toy(60, 1, signal=False)
    out = TR.fit_net(net, tr, tu, 59, "cpu", max_epochs=30, patience=30)
    z = TR.predict_logits(out["net"], tu, "cpu")
    nll = torch.nn.functional.cross_entropy(torch.tensor(z), torch.tensor(tu.y)).item()
    assert abs(nll - out["tune_nll"]) < 1e-5

def test_unlabelled_rows_are_ignored():
    tr, tu = toy(200, 0), toy(50, 1)
    tr.y[:20] = -1; tu.y[:5] = -1
    out = TR.fit_net(net, tr, tu, 59, "cpu", max_epochs=2)    # -1 reaching the loss would raise
    assert np.isfinite(out["tune_nll"]) and not out["net"].training
