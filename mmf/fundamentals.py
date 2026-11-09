"""Point-in-time financial ratios from the annual statements (data/bctc.parquet), one snapshot per (symbol, session)."""
import numpy as np
import pandas as pd

from . import config as C
from nlp_module.forecast.fundamentals import annual_wide

FUND_FEATS = ["ep", "eps_yoy", "roe", "roa", "de", "gross_margin", "rev_yoy", "current_ratio", "has_bctc", "is_bank"]
RATIOS = ["ep", "eps_yoy", "roe", "roa", "de", "gross_margin", "rev_yoy", "current_ratio"]
RATIO_CLIP = 5.0; GROWTH_CLIP = 2.0; MAX_AGE = pd.Timedelta(days=400)   # a missed filing must not carry the year before
ITEMS = {"eps": ["is_lai_co_ban_tren_co_phieu"],
         "np": ["is_lai_lo_thuan_sau_thue", "is_loi_nhuan_sau_thue"],           # non-bank / bank
         "eq": ["bs_von_chu_so_huu_4d280b22", "bs_von_chu_so_huu_6cda78ae"],
         "ta": ["bs_tong_tai_san"],
         "debt": ["bs_no_phai_tra", "bs_tong_no_phai_tra"],
         "rev": ["is_doanh_so_thuan", "is_tong_thu_nhap_hoat_dong"],
         "gp": ["is_lai_gop"], "ca": ["bs_tai_san_ngan_han"], "cl": ["bs_no_ngan_han"]}


def annual_ratios(path=C.BCTC_FILE):
    """-> symbol, year, available_at, eps and the raw (unclipped) ratios, one row per fiscal year."""
    long = pd.read_parquet(path)
    for c in ("ticker", "statement", "item_code"):
        long[c] = long[c].astype(str)           # string[python] == comparisons are all False on pandas 2.3.3
    codes = {c for v in ITEMS.values() for c in v}
    long = long[long["item_code"].isin(codes)]
    # pandas 2.3.3 + Python 3.14 mislabels the pivot index once the frame passes ~32k rows (54 symbols out of 693),
    # so pivot ticker chunks. ponytail: workaround, drop the chunking when pandas is upgraded.
    tk = np.array_split(np.sort(long["ticker"].unique()), 20)
    w = pd.concat([annual_wide(long[long["ticker"].isin(t)]) for t in tk]).sort_values(["symbol", "year"])
    w = w.reset_index(drop=True)
    assert w["symbol"].nunique() == long["ticker"].nunique() and not w.duplicated(["symbol", "year"]).any()
    g = {k: w.reindex(columns=v).bfill(axis=1).iloc[:, 0] for k, v in ITEMS.items()}   # first code present wins

    def yoy(x):                                  # vs the prior fiscal year over |prior|; a gap year is no base
        prev = x.groupby(w["symbol"]).shift().where(w.groupby("symbol")["year"].diff() == 1)
        return (x - prev) / prev.abs()

    w["eps"], w["eps_yoy"], w["rev_yoy"] = g["eps"], yoy(g["eps"]), yoy(g["rev"])
    w["roe"], w["roa"], w["de"] = g["np"] / g["eq"], g["np"] / g["ta"], g["debt"] / g["eq"]
    w["gross_margin"], w["current_ratio"] = g["gp"] / g["rev"], g["ca"] / g["cl"]
    out = w[["symbol", "year", "available_at", "eps", "eps_yoy", "roe", "roa", "de", "gross_margin", "rev_yoy",
             "current_ratio"]]
    return out.replace([np.inf, -np.inf], np.nan)


def add_fundamentals(rows):
    """rows: symbol, as_of_session, close (thousand VND) -> rows + FUND_FEATS from the latest fiscal year known by the
    session (available_at <= session, at most 400 days old). Imputation (explicit): no usable statement (eps, roe, roa
    or de missing) => has_bctc 0 and every ratio 0; other missing ratios => 0. Ratios clipped to +-5, growth to +-2."""
    a = annual_ratios()
    d = rows.assign(_sym=rows["symbol"].astype(str), _i=range(len(rows)))
    a = a.astype({"available_at": d["as_of_session"].dtype})   # merge_asof refuses mixed ns/us resolutions
    m = pd.merge_asof(d.sort_values("as_of_session"), a.rename(columns={"symbol": "_sym"}).sort_values("available_at"),
                      left_on="as_of_session", right_on="available_at", by="_sym", direction="backward",
                      tolerance=MAX_AGE).sort_values("_i")
    m["ep"] = (m["eps"] / (m["close"] * 1000)).replace([np.inf, -np.inf], np.nan)
    ok = np.isfinite(m[["eps", "roe", "roa", "de"]]).all(axis=1)
    m[RATIOS] = m[RATIOS].clip(-RATIO_CLIP, RATIO_CLIP)
    g = ["eps_yoy", "rev_yoy"]
    m[g] = m[g].clip(-GROWTH_CLIP, GROWTH_CLIP)
    m[RATIOS] = m[RATIOS].where(ok, 0.0).fillna(0.0)
    m["has_bctc"] = ok.astype(float)
    m["is_bank"] = m["symbol"].isin(C.BANKS).astype(float)
    return m[list(rows.columns) + FUND_FEATS].set_index(rows.index)
