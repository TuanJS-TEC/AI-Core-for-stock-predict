import json

import numpy as np, pandas as pd, pytest, torch
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier

from mmf import config as C, experiment as EX, explain as X
from mmf.models import FusionNet
from mmf.windows import Samples
from test_mmf_experiment import FOLD, SYMS, fake_frame


def _probs_from_fund_only(s):
    a = s.x_fund[:, 0]
    return softmax(np.c_[-2 * a, np.zeros_like(a), 2 * a], axis=1)


@pytest.fixture
def toy_samples():
    """5 symbols, 40 sessions; the label follows x_fund[:, 0]."""
    r = np.random.default_rng(0)
    n, days = 200, pd.bdate_range("2024-01-01", periods=40).to_numpy()
    fund = r.normal(size=(n, 3)).astype(np.float32)
    y = np.digitize(fund[:, 0], [-0.43, 0.43]).astype(np.int64)
    tick = np.tile(np.arange(5), 40)
    return Samples(r.normal(size=(n, 8, 4)).astype(np.float32), r.normal(size=(n, 3, 2)).astype(np.float32), fund, tick, y,
                   np.array(list("ABCDE"))[tick], np.repeat(days, 5), np.repeat(days, 5))


@pytest.fixture
def toy_net():
    torch.manual_seed(0)
    return FusionNet(("price", "fund", "text"), "attention", 5, 4, 2, 3).eval()


def test_permutation_detects_informative_group(toy_samples):
    t = X.group_permutation(_probs_from_fund_only, toy_samples).set_index("group")
    assert t.loc["fund", "delta_logloss"] > 0.05
    assert t.loc["text", "delta_logloss"] == pytest.approx(0, abs=1e-9)
    assert t.loc["fund", "lo"] <= t.loc["fund", "delta_logloss"] <= t.loc["fund", "hi"]


def test_permutation_stays_within_same_session(toy_samples):
    perm = X._shuffle_group(toy_samples, "fund", np.random.default_rng(0))
    assert not np.array_equal(perm.x_fund, toy_samples.x_fund)
    assert np.array_equal(perm.x_text, toy_samples.x_text) and np.array_equal(perm.ticker, toy_samples.ticker)
    for d in np.unique(toy_samples.as_of):
        k = toy_samples.as_of == d
        assert sorted(map(tuple, perm.x_fund[k])) == sorted(map(tuple, toy_samples.x_fund[k]))


def test_shap_shares_sum_to_one_per_class(toy_net, toy_samples):
    t = X.shap_by_group(toy_net, toy_samples, toy_samples)
    assert np.allclose(t.groupby("class")["share"].sum(), 1)
    assert set(t["group"]) == {"price", "fund", "text"} and t["class"].nunique() == 3
    assert isinstance(toy_net.ticker, torch.nn.Embedding)      # the caller's net is handed back whole


def test_shap_finds_the_only_group_the_net_reads(toy_samples):
    """A net that reads fund only gets (nearly) all the share; ticker embedding is left out of the three groups."""
    net = FusionNet(("price", "fund", "text"), "concat", 5, 4, 2, 3).eval()
    with torch.no_grad():
        for b in ("price", "text"):
            for p in net.enc[b].parameters():
                p.zero_()
    t = X.shap_by_group(net, toy_samples, toy_samples)
    assert (t[t["group"] == "fund"]["share"] > 0.99).all()


def test_lime_explains_the_predicted_class_with_the_informative_feature():
    r = np.random.default_rng(0)
    x = r.normal(size=(400, 3)); y = np.digitize(x[:, 0], [-0.43, 0.43])
    gbm = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(x, y)
    rows = np.array([[2.0, 0, 0], [-2.0, 0, 0], [0.0, 0, 0]])
    cases = X.lime_cases(gbm, x, ["a", "b", "c"], rows)
    assert [c["class"] for c in cases[:2]] == ["up", "down"] and len(cases) == 3
    for c in cases[:2]:
        name, weight = c["weights"][0]
        assert "a" in name and weight > 0 and c["proba"].sum() == pytest.approx(1)


def test_locked_stack_is_explained_through_attention():
    assert X.explained_arm("stack") == "fusion_attention"
    assert X.explained_arm("fusion_concat") == "fusion_concat"


def _priced_bundle():
    f = fake_frame()
    px = np.exp(np.random.default_rng(1).normal(0, 0.01, len(f)).cumsum()) * 10
    b = EX.bundle_from_frame(f.assign(open=px, close=px * 1.001), 5, FOLD, "phobert-foldB", "tinydataset", SYMS)
    return b


def test_cli_report_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "MAX_SHAP", 10)
    b = _priced_bundle()
    run = tmp_path / "run"
    EX.run_all(None, (5,), [b.fold], run, "cpu", arms=("fusion_attention",), seeds=(59,), build=lambda *a: b, max_epochs=1)
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"5": {"fusion": "fusion_attention", "dataset_id": "tinydataset", "picks": {"A": {}}}}))
    out = X.write_report(run, lock, lambda h, name: b, tmp_path / "x.md", tmp_path / "figs")
    text = out.read_text()
    for part in ("T+2", "chưa xác nhận", "Độ phủ", "Permutation", "SHAP", "LIME", "Trọng số attention", "reliability_h5.png"):
        assert part in text
    assert (tmp_path / "figs/reliability_h5.png").stat().st_size and (tmp_path / "figs/confusion_h5.png").stat().st_size


def test_cli_refuses_weights_that_do_not_reproduce_the_run(tmp_path):
    b = _priced_bundle()
    run = tmp_path / "run"
    EX.run_all(None, (5,), [b.fold], run, "cpu", arms=("fusion_attention",), seeds=(59,), build=lambda *a: b, max_epochs=1)
    pt = run / "h5/A/fusion_attention/seed59.pt"
    state = torch.load(pt, map_location="cpu", weights_only=True)
    state["out.weight"][0] += 1.0     # one class only: a shift on all three leaves softmax unchanged
    torch.save(state, pt)
    with pytest.raises(ValueError, match="do not reproduce"):
        X.load_checked_net(b, run, "fusion_attention")
