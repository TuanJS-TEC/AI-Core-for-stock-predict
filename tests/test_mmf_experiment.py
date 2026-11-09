from types import SimpleNamespace

import numpy as np, pandas as pd, pytest, torch
from mmf import baselines as B, config as C, evaluate as E, experiment as X, frame as FR, train as TR
from mmf.windows import BINARY
from src.splits import Fold

SYMS = ["AAA", "BBB", "CCC"]
DAYS = pd.bdate_range("2023-01-02", periods=400)
FOLD = Fold("A", DAYS[199], DAYS[259], DAYS[319], DAYS[379])     # blocks of 200/60/60/60 sessions


def fake_frame():
    """3 symbols, 400 sessions. A text hole in AAA drops windows the row-based baselines would otherwise keep."""
    r = np.random.default_rng(0)
    cols = [c for g in FR.GROUPS.values() for c in g]
    parts = []
    for s in SYMS:
        d = pd.DataFrame(r.normal(size=(400, len(cols))), columns=cols)
        d[BINARY] = r.integers(0, 2, (400, len(BINARY))).astype(float)
        end = pd.Series(DAYS).shift(-5)
        d = d.assign(symbol=s, as_of_session=DAYS, label_end_date=end, label=r.integers(0, 3, 400).astype(float),
                     status=np.where(end.notna(), "ok", "not_matured"))
        d["label"] = d["label"].where(end.notna())
        parts.append(d)
    f = pd.concat(parts, ignore_index=True)
    f["usable"] = f["status"] == "ok"
    f.loc[(f.symbol == "AAA") & (f.as_of_session == DAYS[330]), "sentiment_mean"] = np.nan
    return f


@pytest.fixture
def tiny_bundle():
    return X.bundle_from_frame(fake_frame(), 5, FOLD, "phobert-foldB", "tinydataset", SYMS)


def run(tmp_path, b):
    X.run_all(None, (5,), [b.fold], tmp_path, "cpu", seeds=(59,), build=lambda *a: b, max_epochs=1)


def _eval_keys(f):
    d = pd.read_parquet(f)
    d = d[d["block"].astype(str) == "eval"]
    return set(zip(d["symbol"].astype(str), d["as_of_session"]))


def _logits(b):
    r = np.random.default_rng(1)
    return r.normal(size=(len(b.samples["calib"]), 3)), r.normal(size=(len(b.samples["eval"]), 3))


def test_all_arms_write_same_eval_keys(tmp_path, tiny_bundle):
    run(tmp_path, tiny_bundle)
    files = list(tmp_path.glob("h5/*/*/seed*.parquet"))
    keys = {f.parent.name: _eval_keys(f) for f in files}
    first = next(iter(keys.values()))
    assert set(keys) >= set(C.ARMS) | set(C.BASELINES)
    assert all(k == first for k in keys.values())
    assert (tmp_path / "h5/A/prior_fixed/seed0.parquet").exists() and (tmp_path / "h5/A/dl_price/seed59.pt").exists()


def test_probabilities_are_valid(tmp_path, tiny_bundle):
    run(tmp_path, tiny_bundle)
    d = pd.read_parquet(next(tmp_path.glob("h5/*/fusion_attention/seed59.parquet")))
    p = d[E.PROB_COLS].to_numpy()
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1)


def test_rerun_skips_existing_files(tmp_path, tiny_bundle):
    run(tmp_path, tiny_bundle)
    before = {f: f.stat().st_mtime_ns for f in tmp_path.glob("h5/*/*/seed*.parquet")}
    run(tmp_path, tiny_bundle)
    assert before == {f: f.stat().st_mtime_ns for f in tmp_path.glob("h5/*/*/seed*.parquet")}


def test_eval_labels_do_not_change_calibration(tiny_bundle):
    a = X.calibrate(tiny_bundle, *_logits(tiny_bundle))
    tiny_bundle.samples["eval"].y[:] = 0
    assert a == X.calibrate(tiny_bundle, *_logits(tiny_bundle))


def test_shuffle_changes_only_train_labels(tiny_bundle):
    s = X.shuffle_train_labels(tiny_bundle)
    assert not np.array_equal(s.samples["train"].y, tiny_bundle.samples["train"].y)
    for b in ("tune", "calib", "eval"):
        np.testing.assert_array_equal(s.samples[b].y, tiny_bundle.samples[b].y)


def test_shuffle_uses_one_date_permutation_and_reaches_baselines(tiny_bundle):
    t, s = tiny_bundle.samples["train"], X.shuffle_train_labels(tiny_bundle)
    grid = lambda y: pd.DataFrame({"s": t.symbol, "d": t.as_of, "y": y}).set_index(["d", "s"])["y"]
    old, new = grid(t.y), grid(s.samples["train"].y)
    day = new.index.get_level_values("d")[0]
    source = [d for d in np.unique(t.as_of) if (old.loc[d].to_numpy() == new.loc[day].to_numpy()).all()]
    assert source                                                   # all 3 symbols took their labels from one date
    np.testing.assert_array_equal(s.blocks["train"]["label"].to_numpy(), s.samples["train"].y)


def test_blocks_hold_only_rows_kept_by_windows(tiny_bundle):
    b = tiny_bundle
    for k in ("train", "tune", "calib", "eval"):
        np.testing.assert_array_equal(b.blocks[k].index.to_numpy(), b.samples[k].row_index)
    held = pd.DataFrame({"s": b.samples["eval"].symbol, "d": b.samples["eval"].as_of})
    assert len(held) == 3 * 55 - 10 and not ((held.s == "AAA") & (held.d == DAYS[330])).any()


def test_arm_output_is_calibrated_logits_of_saved_weights(tmp_path, tiny_bundle):
    b, out = tiny_bundle, tmp_path / "x/seed59.parquet"
    X.run_arm(b, "fusion_concat", 59, "gru", out, "cpu", max_epochs=1)
    d = pd.read_parquet(out)
    assert {"symbol", "as_of_session", "label_end_date", "y_true", "block", *E.PROB_COLS, "prior_down", "prior_flat", "prior_up", "alpha",
            "arm", "h", "fold", "seed", "price_kind", "nlp_version", "dataset_id", "git_commit"} <= set(d.columns)
    assert (d.arm.iloc[0], d.h.iloc[0], d.fold.iloc[0], d.seed.iloc[0], d.price_kind.iloc[0]) == ("fusion_concat", 5, "A", 59, "gru")
    assert d.nlp_version.iloc[0] == "phobert-foldB" and d.dataset_id.iloc[0] == "tinydataset" and d.git_commit.iloc[0]
    net = X._net_maker(b, "fusion_concat", "gru")()
    net.load_state_dict(torch.load(out.with_suffix(".pt")))
    z = {k: TR.predict_logits(net, b.samples[k], "cpu") for k in ("calib", "eval")}
    t, alpha = X.calibrate(b, z["calib"], z["eval"])
    for k in z:
        np.testing.assert_allclose(d[d.block.astype(str) == k][E.PROB_COLS].to_numpy(), E.apply_temperature(z[k], t), atol=1e-6)
    assert (d.alpha == alpha).all()


def test_baseline_probabilities_are_written_unchanged(tmp_path, tiny_bundle):
    b, out = tiny_bundle, tmp_path / "prior_fixed/seed0.parquet"
    X.run_baseline(b, "prior_fixed", out)
    d = pd.read_parquet(out)
    p = B.predict_baseline("prior_fixed", b.blocks, b.scaled, b.frame)
    np.testing.assert_array_equal(d[d.block.astype(str) == "eval"][E.PROB_COLS].to_numpy(), p["eval"])
    assert d.seed.iloc[0] == 0 and d.alpha.iloc[0] in E.ALPHAS
    y = b.blocks["train"].query("symbol == 'BBB'")["label"].astype(int)
    row = d[d.symbol.astype(str) == "BBB"].iloc[0]
    assert row[["prior_down", "prior_flat", "prior_up"]].to_numpy(float) == pytest.approx((np.bincount(y, minlength=3) + 1) / (len(y) + 3))


def test_select_takes_lowest_tune_nll_then_first_on_tie_and_reuses_file(tmp_path, tiny_bundle, monkeypatch):
    nll = {"lstm": 1.0, "gru": 0.5, "transformer": 0.5}
    seen = []
    def fake_fit(make_net, *a, **k):
        kind = make_net().enc["price"].kind
        seen.append(kind)
        return {"tune_nll": nll[kind]}
    monkeypatch.setattr(X.TR, "fit_net", fake_fit)
    assert X.select_price_kind(tiny_bundle, tmp_path, "cpu") == "gru"
    assert len(seen) == len(C.PRICE_KINDS) * len(C.SELECT_SEEDS)
    assert (tmp_path / "h5/A/select.json").exists()
    seen.clear(); nll["lstm"] = 0.1
    assert X.select_price_kind(tiny_bundle, tmp_path, "cpu") == "gru" and not seen


def test_price_arms_use_selected_kind(tmp_path, tiny_bundle):
    run(tmp_path, tiny_bundle)
    kind = pd.read_json(tmp_path / "h5/A/select.json", typ="series")["price_kind"]
    assert pd.read_parquet(tmp_path / "h5/A/no_text/seed59.parquet").price_kind.iloc[0] == kind
    assert pd.read_parquet(tmp_path / "h5/A/dl_fund/seed59.parquet").price_kind.iloc[0] == "none"


def test_refuses_without_dataset_id(tmp_path):
    with pytest.raises(ValueError, match="dataset_id"):
        X.run_all(SimpleNamespace(dataset_id=None), (5,), [FOLD], tmp_path, "cpu")
    assert not list(tmp_path.iterdir())


def test_fold_bundle_any_fold_scaler_from_train_only(monkeypatch):
    frame = fake_frame()
    monkeypatch.setattr(X, "build_frame", lambda panel, h, version: frame.copy())
    panel = SimpleNamespace(dataset_id="abc", ohlcv=dict.fromkeys(SYMS))
    b = X.fold_bundle(panel, 5, FOLD._replace(name="F"))
    assert (b.fold.name, b.h, b.nlp_version, b.dataset_id, b.tickers) == ("F", 5, C.FOLD_NLP["F"], "abc", SYMS)
    frame.loc[frame.as_of_session > DAYS[199], "ret_5"] = 1e6       # after the train block: must not move the scaler
    other = X.fold_bundle(panel, 5, FOLD._replace(name="F"))
    i = b.blocks["train"].index
    pd.testing.assert_series_equal(b.scaled.loc[i, "ret_5"], other.scaled.loc[i, "ret_5"])


def test_failed_write_leaves_no_file(tmp_path):
    def boom(tmp):
        tmp.write_text("partial")
        raise OSError
    with pytest.raises(OSError):
        X._atomic(tmp_path / "seed59.parquet", boom)
    assert not (tmp_path / "seed59.parquet").exists()


@pytest.mark.parametrize("argv, kind, folds, kwargs", [
    (["--h", "2", "--folds", "B", "C"], "runs", ["B", "C"], {}),
    (["--smoke"], "smoke", ["A"], {"max_epochs": 1, "seeds": (59,)}),
    (["--shuffle-labels", "--h", "5"], "shuffle", ["C"], {"seeds": (59,)}),
])
def test_cli_modes(monkeypatch, argv, kind, folds, kwargs):
    got = {}
    monkeypatch.setattr(X, "build_panel", lambda stock_dir=None: SimpleNamespace(dataset_id="d" * 64))
    monkeypatch.setattr(X, "run_all", lambda panel, h, f, run_dir, device, **k: got.update(h=h, f=f, dir=run_dir, kw=k))
    X.main(argv + ["--device", "cpu"])
    assert got["dir"] == C.DATA / kind / ("d" * 16) and [f.name for f in got["f"]] == folds
    assert {k: v for k, v in got["kw"].items() if k != "build"} == kwargs
    assert ("build" in got["kw"]) == (kind == "shuffle")
