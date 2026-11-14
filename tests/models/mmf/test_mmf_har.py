import numpy as np
import pandas as pd
import pytest

from ai_core.models.mmf import har as H

CAL = pd.bdate_range("2024-01-01", periods=80)
SYMS = ["AAA", "BBB", "CCC", "DDD"]


def _frames(seed=0):
    r = np.random.default_rng(seed)
    out = {}
    for s in SYMS:
        c = 100 * np.exp(np.cumsum(r.normal(0, 0.02, len(CAL))))
        o = c * np.exp(r.normal(0, 0.005, len(CAL)))
        out[s] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99, "close": c}, index=CAL)
    return out


def _ev(sym="AAA", code="DIV", pub=40, date=50, public_date=None, event_date=None):
    d = CAL[date] if event_date is None else event_date
    return pd.DataFrame({"symbol": [sym], "event_id": ["e1"], "event_code": [code],
                         "public_date": [CAL[pub] if public_date is None else public_date], "record_date": [d],
                         "exright_date": [d if code in ("DIV", "ISS") else pd.NaT], "display_date1": [d]})


def _at(f, sym, i, col):
    return f.loc[(sym, CAL[i]), col]


# --- HAR -----------------------------------------------------------------------------------------------------------

def test_har_is_cross_sectionally_demeaned():
    f = H.har_features(_frames())
    s = f.groupby(level="as_of_session").sum().loc[CAL[30]:]
    assert list(f.columns) == H.HAR_FEATS and np.allclose(s.to_numpy(), 0, atol=1e-9)


def test_har_uses_data_up_to_t_only():
    a, fr = H.har_features(_frames()), _frames()
    fr["AAA"].iloc[41:] *= 1.3
    fr["BBB"].loc[CAL[41], "high"] *= 1.2
    b = H.har_features(fr)
    pd.testing.assert_frame_equal(a.xs(CAL[40], level="as_of_session"), b.xs(CAL[40], level="as_of_session"))


def test_har_rv_w_matches_formula():
    f = _frames()["AAA"]
    raw = H._raw_har(f)
    r = np.log(f["close"] / f["close"].shift(1))
    assert np.isclose(raw.loc[CAL[30], "har_rv_w"], np.log((r.iloc[26:31] ** 2).mean() + H.EPS))
    pk = np.log(f["high"] / f["low"]) ** 2 / (4 * np.log(2))
    assert np.isclose(raw.loc[CAL[30], "har_pk_m"], np.log(pk.iloc[9:31].mean() + H.EPS))
    assert np.isnan(raw.loc[CAL[20], "har_rv_m"])          # 22-session window not full yet


def test_har_demeaned_equals_raw_minus_session_mean():
    fr = _frames()
    f = H.har_features(fr)
    raws = {s: H._raw_har(fr[s]).loc[CAL[50]] for s in SYMS}
    mean = pd.DataFrame(raws).T.mean()
    assert np.allclose(f.loc[("BBB", CAL[50])].to_numpy(), (raws["BBB"] - mean).to_numpy())


# --- events --------------------------------------------------------------------------------------------------------

def _evf(events, h=5):
    return H.event_features(events, CAL, SYMS, h)


def test_event_features_shape_and_no_nan():
    f = _evf(_ev())
    assert list(f.columns) == H.EV_FEATS and len(f) == len(SYMS) * len(CAL) and not f.isna().any().any()


def test_event_known_strictly_before_t():
    f = _evf(_ev(pub=40, date=50))
    assert _at(f, "AAA", 40, "ev_exr_in_h") == 0          # announced at 40: not known at 40
    assert all(_at(f, "AAA", i, "ev_exr_in_h") == 1 for i in range(45, 50))
    g = _evf(_ev(pub=39, date=44))
    assert _at(g, "AAA", 40, "ev_exr_in_h") == 1


def test_event_window_is_t_plus_1_to_t_plus_h():
    f = _evf(_ev(pub=10, date=50), h=2)
    assert [_at(f, "AAA", i, "ev_exr_in_h") for i in (47, 48, 49, 50)] == [0, 1, 1, 0]
    assert (f.xs("BBB", level="symbol")["ev_exr_in_h"] == 0).all()


def test_meeting_uses_display_date1():
    f = _evf(_ev(code="AGME", pub=10, date=50), h=2)
    assert _at(f, "AAA", 48, "ev_mtg_in_h") == 1 and _at(f, "AAA", 48, "ev_exr_in_h") == 0


def test_ev_next_capped_and_one_when_none():
    f = _evf(_ev(pub=10, date=50))
    assert np.isclose(_at(f, "AAA", 47, "ev_next"), 3 / 20)
    assert _at(f, "AAA", 20, "ev_next") == 1.0            # 30 sessions away: capped
    assert _at(f, "AAA", 5, "ev_next") == 1.0             # not announced yet
    assert _at(f, "AAA", 50, "ev_next") == 1.0            # event day itself is not "after t"


def test_ev_new5_counts_announcements_in_t_minus_5_to_t_minus_1():
    ev = pd.concat([_ev(pub=32, date=70), _ev(pub=35, date=75).assign(event_id="e2")], ignore_index=True)
    f = _evf(ev)
    assert [_at(f, "AAA", i, "ev_new5") for i in (32, 33, 36, 41)] == [0, 1, 2, 0]


def test_missing_event_date_counts_only_as_announcement():
    f = _evf(_ev(pub=30, event_date=pd.NaT))
    a = f.xs("AAA", level="symbol")
    assert (a["ev_exr_in_h"] == 0).all() and (a["ev_next"] == 1).all() and a["ev_new5"].sum() == 5


def test_symbol_without_events():
    b = _evf(_ev()).xs("CCC", level="symbol")
    assert (b[["ev_exr_in_h", "ev_mtg_in_h", "ev_new5"]] == 0).all().all() and (b["ev_next"] == 1).all()


def test_event_on_holiday_maps_to_next_session():
    sat = CAL[49] + pd.Timedelta(days=1) if CAL[49].weekday() == 4 else None
    if sat is None:          # pick the first Friday after session 40
        i = next(i for i in range(41, 70) if CAL[i].weekday() == 4)
        sat = CAL[i] + pd.Timedelta(days=1)
    nxt = CAL.searchsorted(sat)                           # the Monday
    f = _evf(_ev(pub=10, event_date=sat), h=1)
    assert _at(f, "AAA", nxt - 1, "ev_exr_in_h") == 1 and _at(f, "AAA", nxt - 2, "ev_exr_in_h") == 0


# --- runner and R9 scoring -----------------------------------------------------------------------------------------
from types import SimpleNamespace

from ai_core.models.mmf import evaluate as E, experiment as X
from tests.models.mmf.test_mmf_experiment import DAYS as XDAYS, FOLD as XFOLD, SYMS as XSYMS, fake_frame


def _bundle(binary=False):
    f = fake_frame()
    if binary:
        f["label"] = (f["label"] > 0).astype(float).where(f["label"].notna())
    return X.bundle_from_frame(f, 5, XFOLD, "phobert-foldB", "tinydataset", XSYMS)


def _extra(b, informative=None, seed=0):
    r = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_frame(b.frame[["symbol", "as_of_session"]].astype({"symbol": str}))
    e = pd.DataFrame(r.normal(size=(len(idx), len(H.HAR_FEATS + H.EV_FEATS))), index=idx, columns=H.HAR_FEATS + H.EV_FEATS)
    if informative:
        e[informative] = b.frame["label"].fillna(0).to_numpy() + r.normal(0, 0.3, len(idx))
    return e


def test_predict_returns_valid_probs_in_block_order():
    b = _bundle()
    p = H.predict(b, _extra(b), H.HAR_FEATS)
    for k in ("calib", "eval"):
        assert p[k].shape == (len(b.blocks[k]), 3) and np.allclose(p[k].sum(axis=1), 1)


def test_predict_scaler_from_train_only():
    b = _bundle()
    e = _extra(b)
    ev = pd.MultiIndex.from_frame(b.blocks["eval"][["symbol", "as_of_session"]].astype({"symbol": str}))
    e2 = e.copy()
    e2.loc[ev] = e2.loc[ev] * 50 + 7
    assert np.allclose(H.predict(b, e, H.HAR_FEATS)["calib"], H.predict(b, e2, H.HAR_FEATS)["calib"])


def test_nested_arm_recovers_informative_feature():
    b = _bundle(binary=True)
    e = _extra(b, informative="ev_new5")
    y = b.samples["eval"].y
    ll = {a: E.row_logloss(y, H.predict(b, e, cols)["eval"]).mean() for a, cols in H.ARMS.items()}
    assert ll["logit_har_ev"] < ll["logit_har"] - 0.1


def _panel():
    r = np.random.default_rng(3)
    ohlc = {}
    for s in XSYMS:
        c = 100 * np.exp(np.cumsum(r.normal(0, 0.02, len(XDAYS))))
        ohlc[s] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c}, index=XDAYS)
    return SimpleNamespace(calendar=XDAYS, ohlcv=ohlc, on_calendar=lambda s: ohlc[s])


def _events():
    return pd.DataFrame({"symbol": ["AAA", "BBB"], "event_id": ["a", "b"], "event_code": ["DIV", "AGME"],
                         "public_date": [XDAYS[300], XDAYS[310]], "record_date": [XDAYS[330], XDAYS[340]],
                         "exright_date": [XDAYS[330], XDAYS[339]], "display_date1": [XDAYS[330], XDAYS[350]]})


def test_run_writes_both_arms_with_same_eval_keys_and_coefficients(tmp_path):
    b = _bundle(binary=True)          # the mag label is binary: coef_ has one row
    H.run(_panel(), _events(), (5,), [XFOLD], tmp_path, build=lambda *a: b)
    keys = []
    for arm in H.ARMS:
        d = pd.read_parquet(tmp_path / f"h5/A/{arm}/seed0.parquet")
        keys.append(set(zip(d.loc[d.block == "eval", "symbol"], d.loc[d.block == "eval", "as_of_session"])))
        assert (tmp_path / f"h5/A/{arm}/coef.json").exists()
    assert keys[0] == keys[1] and len(keys[0]) == len(b.blocks["eval"])


def test_placebo_runs_only_ev_arm_under_its_own_name(tmp_path):
    b, builds = _bundle(binary=True), []
    H.run(_panel(), _events(), (5,), [XFOLD], tmp_path, placebo_seeds=(4, 7), build=lambda *a: builds.append(1) or b)
    assert sorted(p.name for p in (tmp_path / "h5/A").iterdir()) == ["logit_har_ev_p4", "logit_har_ev_p7"]
    assert len(builds) == 1           # one bundle per (h, fold), shared by every placebo seed


def test_placebo_permutation_has_no_fixed_point():
    syms = [f"S{i}" for i in range(15)]
    for seed in range(20):
        m = H.derangement(syms, seed)
        assert sorted(m.values()) == sorted(syms) and all(m[s] != s for s in syms)


def _r9_preds(drop_one=False):
    """6 folds; logit_har_ev knows the label, every other arm is flat."""
    rng, rows = np.random.default_rng(0), []
    for fold in ("A", "B", "C", "D", "E", "Fb"):
        for block in ("calib", "eval"):
            for d in pd.bdate_range("2025-01-01", periods=30):
                for s in "ABCD":
                    y = int(rng.integers(0, 2))
                    for arm in ("logit_har_ev", "logit_har", "prior_adaptive", "prior_fixed", "dl_price", "fusion_concat"):
                        p1 = (0.7 if y else 0.3) if arm == "logit_har_ev" else 0.5
                        rows.append(dict(fold=fold, arm=arm, symbol=s, as_of_session=d, block=block, y_true=y,
                                         p_down=1 - p1, p_flat=p1, p_up=0.0))
    return pd.DataFrame(rows)


def test_r9_rows_and_verdict():
    rows = H.r9(_r9_preds(), 2)
    names = [r["so_sanh"] for r in rows]
    assert names[:2] == ["R9a", "R9b"] and all(n.startswith("kèm") for n in names[2:])
    a = rows[0]
    assert a["verdict"] == "Có bằng chứng" and a["n_folds"] == 6 and a["fold_wins"] == 6
    assert rows[1]["verdict"] == "Chưa chứng minh được"


def test_r9_refuses_mismatched_rows():
    ph = _r9_preds()
    H.check_rows(ph)
    bad = ph.drop(ph[(ph.arm == "logit_har") & (ph.block == "eval") & (ph.fold == "B")].index[:1])
    with pytest.raises(ValueError, match="fold B"):
        H.check_rows(bad)


def test_rare_event_flag_is_not_clipped():
    b = _bundle(binary=True)
    e = _extra(b)
    e["ev_exr_in_h"] = 0.0
    tr = _rows_of(b, "train")
    e.loc[tr[:5], "ev_exr_in_h"] = 1.0                       # prevalence far below 3.85%: z > 5 when set
    _, z = H._fit(b, e, H.ARMS["logit_har_ev"])
    j = H.ARMS["logit_har_ev"].index("ev_exr_in_h")
    assert z["train"][:, j].max() > 5
    assert np.abs(z["train"][:, :len(H.HAR_FEATS)]).max() <= 5


def _rows_of(b, k):
    return pd.MultiIndex.from_frame(b.blocks[k][["symbol", "as_of_session"]].astype({"symbol": str}))


def test_fit_refuses_misaligned_features():
    b = _bundle(binary=True)
    e = _extra(b)
    e.index = e.index.set_levels(e.index.levels[1] + pd.Timedelta(days=1000), level=1)
    with pytest.raises(ValueError, match="HAR"):
        H._fit(b, e, H.HAR_FEATS)


def test_run_refuses_rows_unlike_the_reference_run(tmp_path):
    b = _bundle(binary=True)
    ref = tmp_path / "ref"
    H.run(_panel(), _events(), (5,), [XFOLD], ref, build=lambda *a: b)
    (ref / "h5/A/logit_har").rename(ref / "h5/A/prior_fixed")
    H.run(_panel(), _events(), (5,), [XFOLD], tmp_path / "ok", build=lambda *a: b, ref_dir=ref)
    d = pd.read_parquet(ref / "h5/A/prior_fixed/seed0.parquet")
    d.drop(d[d.block == "eval"].index[:1]).to_parquet(ref / "h5/A/prior_fixed/seed0.parquet")
    with pytest.raises(ValueError, match="reference"):
        H.run(_panel(), _events(), (5,), [XFOLD], tmp_path / "bad", build=lambda *a: b, ref_dir=ref)
