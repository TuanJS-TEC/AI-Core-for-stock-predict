"""Đối chiếu giá đã lưu (nguồn KBS) với TCBS Open API. Chỉ đọc stock/, không bao giờ ghi vào đó.

  ~/.venv_6x/bin/python crosscheck_tcbs.py            mọi mã có trong stock/
  ~/.venv_6x/bin/python crosscheck_tcbs.py ACB FPT    một vài mã

Mỗi mã: tải C.SYNC_OVERLAP_SESSIONS phiên gần nhất từ TCBS, so với parquet trong cùng khoảng ngày. Ghi nối thêm vào
C.CROSSCHECK_FILE mỗi chỗ lệch: giá quá C.SYNC_BASIS_TOL, khối lượng lệch tương đối quá C.CROSSCHECK_VOL_TOL, phiên có
ở bên này mà thiếu ở bên kia, hoặc lỗi tải. Phiên TCBS chưa mở (cần Smart OTP mỗi ngày) thì bỏ qua cả lượt, thoát 0.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from ai_core.prices import config as C
from ai_core.prices.data import load_ohlcv

PRICES = ["open", "high", "low", "close"]
COLUMNS = ["checked_at", "symbol", "session", "field", "stored", "tcbs", "note"]
SESSION_ERRORS = {"OtpRequiredError", "LoginRequiredError", "AuthError", "CredentialError"}   # vnstock_oapi.errors


def compare(stored: pd.DataFrame, fresh: pd.DataFrame, price_tol: float = C.SYNC_BASIS_TOL,
            vol_tol: float = C.CROSSCHECK_VOL_TOL) -> pd.DataFrame:
    """Hai bảng OHLCV có chỉ số là ngày phiên. So trong khoảng [phiên đầu của fresh, phiên cuối của stored]."""
    lo, hi = fresh.index.min(), stored.index.max()
    s, f = stored.loc[lo:hi], fresh.loc[:hi]
    rows = [(d, "missing_in_tcbs", s.at[d, "close"], None) for d in s.index.difference(f.index)]
    rows += [(d, "missing_in_stored", None, f.at[d, "close"]) for d in f.index.difference(s.index)]
    both = s.index.intersection(f.index)
    for col in PRICES + ["volume"]:
        a, b = s.loc[both, col], f.loc[both, col]
        bad = (a - b).abs() > price_tol if col != "volume" else (a - b).abs() > vol_tol * a.abs()
        rows += [(d, col, a[d], b[d]) for d in both[bad.to_numpy()]]
    out = pd.DataFrame(rows, columns=["session", "field", "stored", "tcbs"]).astype({"stored": float, "tcbs": float})
    return out.assign(note="").sort_values(["session", "field"], ignore_index=True)


def fetch_tcbs(symbol: str, start: str, end: str) -> pd.DataFrame:
    from vnstock import Market   # vnstock thế hệ 6, chỉ có trong ~/.venv_6x
    df = Market(C.CROSSCHECK_SOURCE).equity(symbol).ohlcv(start=start, end=end, interval=C.SYNC_INTERVAL)
    t = pd.to_datetime(df["time"])
    t = t.dt.tz_convert(None) if t.dt.tz is not None else t
    return df.set_index(pd.DatetimeIndex(t.dt.normalize(), name="date"))[PRICES + ["volume"]].astype(float)


def run(stock_dir, out_csv, fetch=fetch_tcbs, symbols=None, delay: float = C.SYNC_API_DELAY,
        n: int = C.SYNC_OVERLAP_SESSIONS) -> dict:
    stock_dir, out_csv = Path(stock_dir), Path(out_csv)
    suffix = C.STOCK_FILE.format(ticker="")
    symbols = symbols or sorted(p.name[:-len(suffix)] for p in stock_dir.glob(C.STOCK_FILE.format(ticker="*")))
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    found, errors = [], 0
    for i, sym in enumerate(symbols):
        stored = load_ohlcv(sym, stock_dir)[PRICES + ["volume"]]
        start, end = str(stored.index[-min(n, len(stored))].date()), str(stored.index[-1].date())
        try:
            out = compare(stored, fetch(sym, start, end))
        except Exception as e:
            if type(e).__name__ in SESSION_ERRORS:
                return {"status": "skipped", "reason": f"{type(e).__name__}: {e}"}
            errors += 1
            out = pd.DataFrame({"session": [pd.NaT], "field": ["fetch_error"], "stored": [float("nan")],
                                "tcbs": [float("nan")], "note": [f"{type(e).__name__}: {e}"[:200]]})
        if len(out):
            found.append(out.assign(checked_at=now, symbol=sym))
        if delay and i < len(symbols) - 1:
            time.sleep(delay)
    rows = pd.concat(found, ignore_index=True)[COLUMNS] if found else pd.DataFrame(columns=COLUMNS)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
    return {"status": "ok", "symbols": len(symbols), "differences": int((rows.field != "fetch_error").sum()),
            "errors": errors}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="*")
    a = ap.parse_args(argv)
    res = run(C.STOCK_DIR, C.CROSSCHECK_FILE, symbols=a.symbols or None)
    print(f"{pd.Timestamp.now().isoformat(timespec='seconds')} crosscheck {C.CROSSCHECK_SOURCE}: {res}", flush=True)


if __name__ == "__main__":
    main()
