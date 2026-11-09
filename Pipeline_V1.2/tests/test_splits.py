import numpy as np
import pandas as pd

from src import splits as S

CAL = pd.bdate_range("2021-01-04", periods=800)
LAST = str(CAL[780].date())
BLOCKS = ("train", "tune", "calib", "eval")


def _samples(end=None, lag=5):
    """Mẫu hằng ngày, label_end = as_of + lag phiên; nhãn chưa tới hạn trong dữ liệu cắt ở `end` là NaT."""
    asof = CAL if end is None else CAL[CAL <= end]
    ends = pd.Series(asof).shift(-lag)
    return pd.DataFrame({"as_of_session": asof, "label_end_date": ends.to_numpy()})


def _ranges(f):
    return {"train": (None, f.train_end), "tune": (f.train_end, f.tune_end),
            "calib": (f.tune_end, f.calib_end), "eval": (f.calib_end, f.eval_end)}


def test_make_date_folds_blocks_are_63_sessions_and_eval_disjoint():
    a, b, c = S.make_date_folds(CAL, LAST)
    assert [f.name for f in (a, b, c)] == ["A", "B", "C"]
    assert c.eval_end == pd.Timestamp(LAST)
    for f in (a, b, c):
        pos = [CAL.get_loc(d) for d in f[1:]]
        assert np.diff(pos).tolist() == [63, 63, 63]
    assert b.eval_end == c.calib_end and a.eval_end == b.calib_end


def test_purge_no_train_label_crosses_cutoff():
    s = _samples()
    for f in S.make_date_folds(CAL, LAST):
        out = S.split_samples(s, f)
        for name, (prev, end) in _ranges(f).items():
            blk = out[name]
            assert len(blk) > 0
            assert blk["label_end_date"].max() <= end
            if prev is not None:
                assert blk["as_of_session"].min() > prev
        near = CAL[CAL.get_loc(f.train_end) - 2]
        assert near not in set(out["train"]["as_of_session"])


def test_blocks_are_disjoint_and_cover_only_kept():
    s = _samples()
    folds = S.make_date_folds(CAL, LAST)
    for f in folds:
        out = S.split_samples(s, f)
        idx = [set(out[k].index) for k in BLOCKS]
        assert sum(map(len, idx)) == len(set().union(*idx))
    m = S.split_manifest(s, folds)
    assert list(m.columns) == ["fold", "block", "start", "end", "n_kept", "n_purged"]
    assert len(m) == 12
    for _, r in m.iterrows():
        inside = s["as_of_session"] <= r["end"]
        if pd.notna(r["start"]):
            inside &= s["as_of_session"] > r["start"]
        assert r["n_kept"] + r["n_purged"] == inside.sum()
        assert r["n_purged"] > 0


def test_appending_new_data_does_not_move_folds():
    folds = S.make_date_folds(CAL, LAST)
    short, long = _samples(end=folds[-1].eval_end), _samples()
    for f in folds:
        a, b = S.split_samples(short, f), S.split_samples(long, f)
        for k in BLOCKS:
            assert a[k].index.equals(b[k].index)


def test_config_folds_are_ordered_and_end_at_frozen_test_end():
    folds = S.folds_from_config()
    assert len(folds) == 3
    for f in folds:
        assert list(f[1:]) == sorted(f[1:]) and len(set(f[1:])) == 4
    assert all(a.eval_end < b.eval_end and a.train_end < b.train_end for a, b in zip(folds, folds[1:]))
    assert folds[-1].eval_end == pd.Timestamp("2026-10-30")
