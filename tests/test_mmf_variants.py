from dataclasses import replace

import numpy as np, pandas as pd, pytest
from src import config as PC
from src.data import build_panel, load_raw
from src.price_features import FEATURES
from src.splits import folds_from_config, split_samples
from mmf import config as C, experiment as X, frame as FR
from mmf.aux import Aux
from mmf.context import CTX_FEATS
from mmf.text import TEXT_FEATS
from mmf.windows import Scalers, make_samples

H = 5
FOLD = {f.name: f for f in folds_from_config()}["C"]


def no_text(version, symbols, sessions):
    idx = pd.MultiIndex.from_product([symbols, sessions], names=["symbol", "as_of_session"])
    return pd.DataFrame(0.0, index=idx, columns=TEXT_FEATS).reset_index()


@pytest.fixture(scope="module")
def panel():
    return build_panel(PC.TICKERS, load_raw())


@pytest.fixture(scope="module")
def aux(panel):
    """30 fake external symbols (scaled copies of real ones, first 100 sessions absent), an index and four global walks."""
    own, cal = sorted(panel.ohlcv), panel.calendar
    frames = {}
    for i in range(30):
        f = panel.on_calendar(own[i % len(own)]).copy()
        f[["open", "high", "low", "close"]] *= 1 + 0.01 * (i + 1)
        f.iloc[:100] = np.nan
        frames[f"X{i:02d}"] = f
    index = pd.concat([panel.on_calendar(s) for s in own]).groupby(level=0).mean()
    days = pd.bdate_range(cal[0] - pd.Timedelta(days=60), cal[-1])
    r = np.random.default_rng(0)
    glob = pd.concat([pd.DataFrame({"series": s, "date": days, "close": 50 * np.exp(r.normal(0, 0.01, len(days)).cumsum())})
                      for s in C.GLOBAL_SERIES], ignore_index=True)
    return Aux(frames, index, glob, "test")


@pytest.fixture(scope="module")
def bundles(panel, aux):
    with pytest.MonkeyPatch.context() as mp:
        frame = FR.build_frame
        mp.setattr(X, "build_frame", lambda p, h, v: frame(p, h, v, text_fn=no_text))
        out = {v: X.variant_bundle(panel, aux, H, FOLD, v) for v in C.VARIANTS}
        out["v1"] = X.fold_bundle(panel, H, FOLD)
    return out


def keys(s):
    return list(zip(s.symbol, s.as_of))


def test_v1_samples_unchanged(panel, bundles):
    frame = FR.build_frame(panel, H, "x", text_fn=no_text)
    blocks = split_samples(frame[frame["usable"]], FOLD)
    scaled = Scalers.fit(blocks["train"]).apply(frame)
    b = bundles["v1"]
    assert b.price_cols is None and b.variant is None and b.aux_id is None
    for k in ("train", "eval"):
        want = make_samples(scaled, blocks[k], sorted(panel.ohlcv))
        assert np.array_equal(b.samples[k].x_price, want.x_price) and np.array_equal(b.samples[k].y, want.y)
        assert keys(b.samples[k]) == keys(want)


def test_ext_rows_only_in_train(bundles):
    b = bundles["ext"]
    assert any(s.startswith("X") for s in b.samples["train"].symbol)
    for k in ("tune", "calib", "eval"):
        assert set(b.samples[k].symbol) <= set(PC.TICKERS) and len(b.samples[k])
        assert set(b.blocks[k]["symbol"].astype(str)) <= set(PC.TICKERS)
    assert b.tickers[:len(PC.TICKERS)] == sorted(PC.TICKERS) and len(b.tickers) == len(PC.TICKERS) + 30


def test_ext_eval_rows_equal_v1_eval_rows(bundles):
    for v in ("base", "ext"):
        for k in ("tune", "calib", "eval"):
            assert keys(bundles[v].samples[k]) == keys(bundles["v1"].samples[k])


def test_base_variant_matches_v1_inputs(bundles):
    b = bundles["base"]
    assert np.array_equal(b.samples["eval"].x_price, bundles["v1"].samples["eval"].x_price)
    assert b.variant == "base" and b.aux_id == "test" and b.price_cols == FEATURES


def test_ctx_widens_price_input(bundles):
    b = bundles["ext_ctx"]
    assert b.price_cols == [*FEATURES, *CTX_FEATS]
    for k in ("train", "eval"):
        x = b.samples[k].x_price
        assert x.shape[1:] == (C.LP, len(FEATURES) + len(CTX_FEATS)) and np.isfinite(x).all() and len(x)
    assert set(keys(b.samples["eval"])) <= set(keys(bundles["v1"].samples["eval"]))


def test_ext_train_labels_are_purged(bundles):
    tr = bundles["ext"].samples["train"]
    ext = np.array([s.startswith("X") for s in tr.symbol])
    assert pd.to_datetime(tr.label_end[ext]).max() <= FOLD.train_end and (tr.y[ext] >= 0).all()


def run(tmp_path, b, mp):
    mp.setitem(C.V2_BASELINES, "ext", ("logit_price", "gbm_price"))
    mp.setattr(C, "PRICE_KINDS", ("gru",))
    mp.setattr(C, "SELECT_SEEDS", (59,))
    X.run_variants(None, None, (H,), [b.fold], tmp_path, "cpu", variants=("ext",), seeds=(59,), max_epochs=1, build=lambda *a: b)


def test_out_name_and_paths(tmp_path, bundles, monkeypatch):
    run(tmp_path, bundles["ext"], monkeypatch)
    base = tmp_path / f"h{H}" / "C"
    assert (base / "select@ext.json").exists() and not (base / "select.json").exists()
    for name, seed in (("logit_price", 0), ("gbm_price", 0), ("dl_price", 59)):
        d = pd.read_parquet(base / f"{name}@ext" / f"seed{seed}.parquet")
        assert set(d["arm"]) == {f"{name}@ext"} and set(d["aux_id"]) == {"test"}
        assert set(d["symbol"].astype(str)) <= set(PC.TICKERS)
        p = d[["p_down", "p_flat", "p_up"]].to_numpy()
        assert np.isfinite(p).all() and np.allclose(p.sum(1), 1)


def test_reuse_refuses_other_aux(tmp_path, bundles, monkeypatch):
    run(tmp_path, bundles["ext"], monkeypatch)
    with pytest.raises(ValueError, match="another run"):
        run(tmp_path, replace(bundles["ext"], aux_id="khac"), monkeypatch)
