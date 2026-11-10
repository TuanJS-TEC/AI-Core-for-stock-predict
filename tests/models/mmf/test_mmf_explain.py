import json
from types import SimpleNamespace

import numpy as np, pandas as pd, pytest, torch
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_core.models.mmf import config as C, experiment as EX, explain as X, recommend as R
from ai_core.models.mmf.models import FusionNet
from ai_core.models.mmf.windows import Samples
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
    for part in ("T+2", "chưa xác nhận", "Độ phủ", "Permutation", "SHAP", "LIME", "Trọng số attention", "reliability_h5.png", "calib_coverage", "calib_n", "calib âm"):
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


def _flat_market(bump_day=None):
    """5 symbols, 200 flat sessions (every trade earns -cost); calib ends at session 119, eval at 179, h = 2."""
    days = pd.bdate_range("2024-01-01", periods=200)
    f = pd.DataFrame({"symbol": np.tile(list("ABCDE"), 200), "as_of_session": np.repeat(days, 5), "open": 10.0, "close": 10.0})
    if bump_day is not None:
        f.loc[f.as_of_session == days[bump_day], "close"] = 30.0
    fold = SimpleNamespace(name="A", calib_end=days[119], eval_end=days[179])
    return SimpleNamespace(frame=f, h=2, fold=fold), days


def _block_preds(days, start, stop, block, wide_from):
    """Rows as split_samples keeps them (t + h inside the block); the last sessions have the widest p_up - p_down."""
    d = pd.DataFrame({"symbol": np.tile(list("ABCDE"), stop - start), "as_of_session": np.repeat(days[start:stop], 5), "block": block})
    return d.assign(fold="A", arm="a", p_down=0.3, p_up=np.where(d.as_of_session >= days[wide_from], 0.7, 0.35))


@pytest.mark.parametrize("bump_day", [120, 180])     # first session of eval, first session after eval
def test_margin_and_scores_read_no_price_after_their_block(bump_day):
    """The rows of t = 117 and t = 177 would exit on sessions 120 and 180. A price there must change nothing."""
    b, days = _flat_market()
    preds = pd.concat([_block_preds(days, 60, 118, "calib", 108), _block_preds(days, 120, 178, "eval", 168)], ignore_index=True)
    base, _ = X._simulations(preds, b, "a")
    moved, _ = X._simulations(preds, _flat_market(bump_day)[0], "a")
    keys = ["m", "calib_mean_net", "calib_coverage", "calib_n", "mean_net", "coverage", "n"]
    assert {k: moved[k] for k in keys} == {k: base[k] for k in keys}
    assert base["m"] == 0.0 and base["calib_n"] == 57 * 5 and base["n"] == 57 * 5     # t = 117 and t = 177 are out


def _section_tables(mean, lo, hi, rand):
    rows = [{"h": 2, "fold": f, "pham_vi": "theo luật", "coverage": 0.2, "n": 100, "n_days": 40, "mean_net": mean, "ci_lo": lo, "ci_hi": hi, "rand_median": rand}
            for f in ("A", "gộp")] + [{"h": 2, "fold": "gộp", "pham_vi": "toàn mẫu", "coverage": 1.0, "n": 500, "n_days": 60, "mean_net": -0.01}]
    perm = pd.DataFrame({"h": 2, "fold": list("ABC"), "group": "fund", "delta_logloss": [0.001, 0.002, 0.003], "lo": [0.0005, 0.001, -0.001], "hi": [0.002, 0.003, 0.004]})
    return pd.DataFrame(rows), perm, pd.DataFrame({"h": 2, "group": ["fund"], "share": [1.0]})


def test_conclusion_states_a_loss_plainly():
    text = "\n".join(X._h_section(2, "fusion_attention", *_section_tables(-0.005, -0.009, -0.001, -0.008)))
    assert "cả CI nằm dưới 0" in text and "-0.0050" in text
    assert "một seed, CI chưa hiệu chỉnh" in text and "fund (A +0.0010, B +0.0020, C +0.0030)" in text and "tăng rõ" not in text


def test_conclusion_names_the_fold_worse_than_random_with_its_coverage():
    text = "\n".join(X._h_section(2, "fusion_attention", *_section_tables(-0.009, -0.02, 0.01, -0.005)))
    assert "cả CI nằm dưới 0" not in text and "CI chứa 0" in text
    assert "A (-0.0090, độ phủ 20.0%, 100 lệnh, ngẫu nhiên -0.0050)" in text


def test_calibration_says_when_p_up_is_overstated():
    ev = pd.DataFrame({"p_up": [0.6, 0.6, 0.2], "p_down": [0.2, 0.2, 0.5], "y_true": [0, 1, 0], "m": 0.1, "net_ret": [0.01, np.nan, 0.01]})
    over, line = X._calibration(ev, 2)
    assert over and "1 lệnh của luật: p_up 0.600, tần suất Tăng 0.000" in line
    assert not X._calibration(ev.assign(y_true=[2, 2, 2]), 2)[0]


def test_label_shares_and_coverage_share_one_base():
    ev = pd.DataFrame({"p_up": [0.6, 0.6, 0.2, 0.3], "p_down": [0.2, 0.2, 0.5, 0.3], "m": 0.1, "net_ret": [0.01, np.nan, 0.01, 0.01],
                       "as_of_session": pd.bdate_range("2024-01-01", periods=4)})
    k = X._label_shares(ev)
    assert R.simulate(ev, 0.1, 2)["coverage"] + k["share_tranh"] + k["share_khong_du"] == pytest.approx(1)


def test_conclusion_lists_groups_whose_permutation_lowers_logloss():
    sim, perm, shares = _section_tables(-0.005, -0.009, -0.001, -0.008)
    perm = pd.concat([perm, pd.DataFrame({"h": 2, "fold": list("ABC"), "group": "text", "delta_logloss": [0.001, -0.002, -0.0061],
                       "lo": [-0.001, -0.004, -0.012], "hi": [0.003, 0.0001, -0.0001]})])
    text = "\n".join(X._h_section(2, "fusion_attention", sim, perm, shares))
    assert "Nhóm làm log-loss giảm khi hoán vị (hi < 0; một seed, CI chưa hiệu chỉnh đa so sánh): text (C -0.0061, hi -0.0001)." in text
    assert "giảm khi hoán vị (hi < 0; một seed, CI chưa hiệu chỉnh đa so sánh): không có" in "\n".join(X._h_section(2, "fusion_attention", sim, perm.query("group == 'fund'"), shares))
