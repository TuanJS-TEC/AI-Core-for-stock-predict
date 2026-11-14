import numpy as np, pandas as pd, pytest
from ai_core.models.mmf import fund_quarter as F

CAL = pd.bdate_range("2026-01-01", "2026-11-13")


def rows(sym, *days, close=50.0):
    return pd.DataFrame({"symbol": sym, "as_of_session": pd.to_datetime(list(days)), "close": close})


def test_real_values_and_bank_gaps():
    q = F.quarter_ratios().set_index(["symbol", "year", "quarter"])
    fpt, vcb = q.loc[("FPT", 2026, 3)], q.loc[("VCB", 2026, 3)]
    assert fpt.public_date == pd.Timestamp("2026-10-29")
    assert fpt.np_ttm == pytest.approx(1.028760e13, rel=1e-4) and fpt.q_roe == pytest.approx(0.234, abs=1e-3)
    assert np.isnan(vcb.q_gross_margin) and np.isnan(vcb.q_current_ratio) and np.isfinite(vcb.q_roe)
    assert q.loc[("FPT", 2018, 3), "np_ttm"] != q.loc[("FPT", 2018, 3), "np_ttm"]      # fewer than four quarters: NaN, not a partial sum


def test_quarter_not_visible_on_its_publication_day():
    out = F.add_quarter(rows("FPT", "2026-10-29", "2026-10-30", "2026-11-05", "2026-11-06"), CAL)
    on_day, after, last_fresh, stale = (out.iloc[i] for i in range(4))
    assert on_day.q_roe == pytest.approx(0.244, abs=1e-3) and on_day.q_fresh == 0          # still the quarter before
    assert after.q_roe == pytest.approx(0.234, abs=1e-3) and after.q_fresh == 1
    assert after.q_np_yoy_fresh == after.q_np_yoy != 0
    assert last_fresh.q_fresh == 1 and stale.q_fresh == 0 and stale.q_np_yoy_fresh == 0   # 5 sessions: 30/10 .. 05/11


def test_stale_and_unknown_are_flagged_not_filled():
    stale = F.add_quarter(rows("FPT", "2027-06-01"), pd.bdate_range("2026-01-01", "2027-06-30")).iloc[0]
    unknown = F.add_quarter(rows("XXX", "2026-06-02"), CAL).iloc[0]
    for r in (stale, unknown):
        assert r.q_has_bctc == 0 and r.q_ep == 0 and r.q_roe == 0 and r.q_fresh == 0


def test_nan_free_bank_flag_and_row_order():
    r = pd.concat([rows("VCB", "2026-06-02"), rows("SSI", "2026-06-02"), rows("FPT", "2026-06-02")]).set_axis([7, 3, 5])
    out = F.add_quarter(r, CAL)
    assert list(out.index) == [7, 3, 5] and out["symbol"].tolist() == ["VCB", "SSI", "FPT"]
    assert out[F.QFUND_FEATS].notna().all().all() and out["q_is_bank"].tolist() == [1.0, 0.0, 0.0]
    assert out.loc[7, "q_gross_margin"] == 0 and out.loc[3, "q_has_bctc"] == 1               # SSI now has statements


def test_bundle_with_quarterly_columns_trains_and_baselines_read_them(tmp_path):
    from dataclasses import replace
    from ai_core.models.mmf import baselines as B, experiment as X
    from ai_core.models.mmf.windows import make_samples
    from tests.models.mmf.test_mmf_experiment import FOLD, SYMS, fake_frame
    f = fake_frame()
    f[F.QFUND_FEATS] = np.random.default_rng(0).normal(size=(len(f), len(F.QFUND_FEATS)))
    b = X.bundle_from_frame(f, 5, FOLD, "phobert-foldB", "tinydataset", SYMS)
    q = replace(b, samples={k: make_samples(b.scaled, rows, SYMS, fund_cols=F.QFUND_FEATS) for k, rows in b.blocks.items()},
                fund_cols=list(F.QFUND_FEATS))
    assert q.samples["eval"].x_fund.shape[1] == len(F.QFUND_FEATS) and b.samples["eval"].x_fund.shape[1] != len(F.QFUND_FEATS)
    assert (q.samples["eval"].as_of == b.samples["eval"].as_of).all()                 # same rows as the annual bundle
    X.run_arm(q, "dl_fund", 59, "lstm", tmp_path / "seed59.parquet", "cpu", max_epochs=1)
    p_q = B.predict_baseline("logit_all", q.blocks, q.scaled, q.frame, fund_cols=q.fund_cols)["eval"]
    assert not np.allclose(p_q, B.predict_baseline("logit_all", b.blocks, b.scaled, b.frame)["eval"])
