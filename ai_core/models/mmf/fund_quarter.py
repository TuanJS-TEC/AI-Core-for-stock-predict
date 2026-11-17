"""Point-in-time quarterly financial ratios (prereg addendum R12, 2026-11-14) from the VCI raw statements, which carry the
real publication date of every quarter (annual_ratios has fiscal years and an assumed 1 April).

Files (C.BCTC_Q_DIR): bctc_quarter_vci_income_statement.parquet, bctc_quarter_vci_balance_sheet.parquet, one row per
(ticker, yearReport, lengthReport). Refetch (vnstock 4, .venv): python -m ai_core.models.mmf.fund_quarter
Known limit: values are the source's current ones; a quarter restated after its publication carries the restated numbers.
"""
import numpy as np
import pandas as pd

from . import config as C

RATIOS = ["q_ep", "q_np_yoy", "q_np_accel", "q_roe", "q_roa", "q_de", "q_gross_margin", "q_rev_yoy", "q_current_ratio"]
QFUND_FEATS = [*RATIOS, "q_np_yoy_fresh", "q_fresh", "q_has_bctc", "q_is_bank"]
QBINARY = ["q_fresh", "q_has_bctc", "q_is_bank"]
RATIO_CLIP = 5.0; DE_CLIP = 20.0; GROWTH_CLIP = 2.0
MAX_AGE = pd.Timedelta(days=150)     # one reporting cycle plus a late filing: a missed quarter must not carry the one before
FRESH = 5                            # sessions a statement counts as "just published"
# VCI field codes: net sales, gross profit, profit attributable to the parent; current assets, total assets, liabilities,
# current liabilities, equity, paid-in capital (par 10,000 VND -> shares)
IS = {"isa3": "rev", "isa5": "gp", "isa22": "np"}
BS = {"bsa1": "ca", "bsa53": "ta", "bsa54": "debt", "bsa55": "cl", "bsa78": "eq", "bsa80": "paid_in"}
KEY = ["ticker", "yearReport", "lengthReport"]


def quarter_ratios(d=None):
    """-> symbol, year, quarter, public_date, np_ttm, shares and the raw ratios, one row per reported quarter.

    TTM sums and year-on-year changes need the four (or the same) quarters to be present: a gap gives NaN, never a
    figure built across it. Banks have no net sales, gross profit or current items: those ratios are NaN for them."""
    d = C.BCTC_Q_DIR if d is None else d
    i = pd.read_parquet(d / "bctc_quarter_vci_income_statement.parquet")[[*KEY, "publicDate", *IS]].rename(columns=IS)
    b = pd.read_parquet(d / "bctc_quarter_vci_balance_sheet.parquet")[[*KEY, *BS]].rename(columns=BS)
    q = i.merge(b, on=KEY, validate="one_to_one").sort_values(KEY).reset_index(drop=True)
    q["public_date"] = pd.to_datetime(q["publicDate"], format="ISO8601", utc=True).dt.tz_localize(None).dt.normalize()
    bank = q["ticker"].isin(C.BANKS)
    q.loc[bank, ["rev", "gp", "ca", "cl"]] = np.nan
    g, qi = q.groupby("ticker"), q["yearReport"] * 4 + q["lengthReport"]
    gap = qi.groupby(q["ticker"]).diff
    q["np_ttm"] = g["np"].rolling(4).sum().reset_index(level=0, drop=True).where(gap(3) == 3)

    def yoy(c):
        prev = g[c].shift(4).where(gap(4) == 4)
        return (q[c] - prev) / prev.abs()

    q["q_np_yoy"], q["q_rev_yoy"] = yoy("np"), yoy("rev")
    q["q_np_accel"] = q["q_np_yoy"].clip(-GROWTH_CLIP, GROWTH_CLIP) - q.groupby("ticker")["q_np_yoy"].shift().clip(-GROWTH_CLIP, GROWTH_CLIP).where(gap(1) == 1)
    q["q_roe"], q["q_roa"], q["q_de"] = q["np_ttm"] / q["eq"], q["np_ttm"] / q["ta"], q["debt"] / q["eq"]
    q["q_gross_margin"], q["q_current_ratio"] = q["gp"] / q["rev"], q["ca"] / q["cl"]
    q["shares"] = q["paid_in"] / 10_000
    out = q.rename(columns={"ticker": "symbol", "yearReport": "year", "lengthReport": "quarter"})
    return out[["symbol", "year", "quarter", "public_date", "np_ttm", "shares", *RATIOS[1:]]].replace([np.inf, -np.inf], np.nan)


def add_quarter(rows, calendar=None, ratios=None):
    """rows: symbol, as_of_session, close (thousand VND) -> rows + QFUND_FEATS from the latest quarter published strictly
    before the session's date (a filing may come out after the close) and at most MAX_AGE old.

    q_fresh = 1 for the first FRESH sessions (of `calendar`, default the sessions in rows) a statement is usable;
    q_np_yoy_fresh = q_np_yoy on those sessions, else 0. Imputation as add_fundamentals: no usable statement (np_ttm,
    roe, roa or de missing) => q_has_bctc 0 and every ratio 0; other missing ratios => 0."""
    a = (quarter_ratios() if ratios is None else ratios).rename(columns={"symbol": "_sym"})
    a["available_at"] = a["public_date"] + pd.Timedelta(days=1)
    d = rows.assign(_sym=rows["symbol"].astype(str), _i=range(len(rows)))
    a = a.astype({"available_at": d["as_of_session"].dtype}).sort_values("available_at")
    m = pd.merge_asof(d.sort_values("as_of_session"), a, left_on="as_of_session", right_on="available_at", by="_sym",
                      direction="backward", tolerance=MAX_AGE).sort_values("_i")
    m["q_ep"] = (m["np_ttm"] / (m["close"] * 1000 * m["shares"])).replace([np.inf, -np.inf], np.nan)
    ok = np.isfinite(m[["np_ttm", "q_roe", "q_roa", "q_de"]]).all(axis=1)
    de = m["q_de"].clip(-DE_CLIP, DE_CLIP)
    m[RATIOS] = m[RATIOS].clip(-RATIO_CLIP, RATIO_CLIP)
    grow = ["q_np_yoy", "q_rev_yoy"]
    m[grow] = m[grow].clip(-GROWTH_CLIP, GROWTH_CLIP)
    m["q_de"] = de
    m[RATIOS] = m[RATIOS].where(ok, 0.0).fillna(0.0)
    cal = pd.DatetimeIndex(np.sort(rows["as_of_session"].unique()) if calendar is None else calendar)
    age = cal.searchsorted(m["as_of_session"]) - cal.searchsorted(m["available_at"].fillna(cal[0]))   # sessions since first usable
    m["q_fresh"] = (ok & (age < FRESH)).astype(float)
    m["q_np_yoy_fresh"] = m["q_np_yoy"] * m["q_fresh"]
    m["q_has_bctc"] = ok.astype(float)
    m["q_is_bank"] = m["symbol"].astype(str).isin(C.BANKS).astype(float)
    return m[list(rows.columns) + QFUND_FEATS].set_index(rows.index)


def fetch(out=None, delay=3.5):
    """Raw quarterly statements of C.TICKERS from VCI (vnstock 4 keeps 4 quarters unless the raw report is asked for)."""
    import time
    import vnstock.explorer.vci.financial as V
    out = C.BCTC_Q_DIR if out is None else out
    for rt in ("income_statement", "balance_sheet"):
        parts = []
        for s in C.TICKERS:
            parts.append(V.Finance(symbol=s, period="quarter", show_log=False)._get_report(report_type=rt, mode="raw", limit=10_000))
            time.sleep(delay)                                   # guest limit: 20 calls a minute
        d = pd.concat(parts, ignore_index=True).dropna(axis=1, how="all")
        d.to_parquet(out / f"bctc_quarter_vci_{rt}.parquet")
        print(rt, d.shape, int(d["yearReport"].min()), int(d["yearReport"].max()), flush=True)


if __name__ == "__main__":
    fetch()
