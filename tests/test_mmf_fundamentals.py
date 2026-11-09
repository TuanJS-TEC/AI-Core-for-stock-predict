import numpy as np, pandas as pd, pytest
from mmf import fundamentals as F

def row(df, s, y): return df[(df.symbol == s) & (df.year == y)].iloc[0]

def test_real_values_nonbank_and_bank():
    a = F.annual_ratios()
    fpt, vcb = row(a, "FPT", 2024), row(a, "VCB", 2024)
    assert fpt.eps == pytest.approx(4292)
    assert fpt.roe == pytest.approx(9.427423e12 / 3.572754e13, rel=1e-3)
    assert fpt.gross_margin == pytest.approx(2.369835e13 / 6.284879e13, rel=1e-3)
    assert fpt.de == pytest.approx(3.627246e13 / 3.572754e13, rel=1e-3)
    assert fpt.current_ratio == pytest.approx(4.553594e13 / 3.483618e13, rel=1e-3)
    assert fpt.available_at == pd.Timestamp("2025-04-01")
    assert vcb.eps == pytest.approx(5571)
    assert vcb.roe == pytest.approx(3.385312e13 / 1.962092e14, rel=1e-3)
    assert vcb.de == pytest.approx(1.889664e15 / 1.962092e14, rel=1e-3)
    assert np.isnan(vcb.gross_margin) and np.isnan(vcb.current_ratio)

def rows(sym, day, close=100.0):
    return pd.DataFrame({"symbol": [sym], "as_of_session": [pd.Timestamp(day)], "close": [close]})

def test_fiscal_year_not_visible_before_available_at():
    before = F.add_fundamentals(rows("FPT", "2025-03-31")).iloc[0]
    after = F.add_fundamentals(rows("FPT", "2025-04-01")).iloc[0]
    assert after.ep == pytest.approx(4292 / (100.0 * 1000))
    assert before.ep != after.ep

def test_stale_year_dropped_and_missing_symbol_flagged():
    stale = F.add_fundamentals(rows("FPT", "2031-01-02")).iloc[0]
    ssi = F.add_fundamentals(rows("SSI", "2025-06-02")).iloc[0]
    for r in (stale, ssi):
        assert r.has_bctc == 0 and r.ep == 0 and r.roe == 0

def test_bank_flag_and_nan_free():
    out = F.add_fundamentals(pd.concat([rows("VCB", "2025-06-02"), rows("FPT", "2025-06-02")]))
    assert out.set_index("symbol").is_bank.to_dict() == {"VCB": 1.0, "FPT": 0.0}
    assert out[F.FUND_FEATS].notna().all().all()

def test_row_order_and_index_kept():
    r = pd.concat([rows("VCB", "2025-06-02"), rows("FPT", "2024-06-03"), rows("FPT", "2025-06-02")])
    r.index = [10, 11, 12]
    out = F.add_fundamentals(r)
    assert list(out.index) == [10, 11, 12] and out.symbol.tolist() == ["VCB", "FPT", "FPT"]
    assert out.has_bctc.tolist() == [1, 1, 1] and out.ep.iloc[1] != out.ep.iloc[2]

def test_bank_leverage_not_flattened():
    assert F.add_fundamentals(rows("VCB", "2025-06-02")).iloc[0].de == pytest.approx(1.889664e15 / 1.962092e14, rel=1e-3)
