"""Auxiliary data of the v2 variants: external training tickers, the VNINDEX and global series, frozen under an aux_id.

Use: python -m ai_core.models.mmf.aux index    (re)download the VNINDEX into C.EXT_DIR
     python -m ai_core.models.mmf.aux freeze   copy C.EXT_DIR + C.GLOBAL_FILE into C.AUX/<aux_id[:16]> and print the path
"""
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from ai_core.prices import config as PC
from ai_core.prices import snapshot as SN
from ai_core.prices.data import load_ohlcv

OHLCV = [*PC.PRICE_COLS, PC.VOLUME_COL]
GLOBAL = "global.parquet"


def _on_calendar(sym, ext_dir, calendar):
    """OHLCV of one symbol on the calendar; an absent session is a NaN row, never filled."""
    return load_ohlcv(sym, ext_dir)[OHLCV].astype(np.float64).reindex(calendar)


def load_frames(ext_dir, calendar, exclude=C.TICKERS):
    """symbol -> OHLCV on the calendar for every stock file in ext_dir, minus `exclude`, the index and short histories."""
    pre, suf = PC.STOCK_FILE.split("{ticker}")
    syms = sorted(p.name[len(pre):-len(suf)] for p in Path(ext_dir).glob(PC.STOCK_FILE.format(ticker="*")))
    frames = {s: _on_calendar(s, ext_dir, calendar) for s in syms if s not in exclude and s != C.INDEX}
    return {s: f for s, f in frames.items() if np.isfinite(f["close"]).sum() >= C.EXT_MIN_SESSIONS}


def load_index(ext_dir, calendar):
    if not (Path(ext_dir) / PC.STOCK_FILE.format(ticker=C.INDEX)).exists():
        raise ValueError(f"thiếu {C.INDEX} trong {ext_dir}")
    return _on_calendar(C.INDEX, ext_dir, calendar)


def _fetch_index():
    from vnstock import Quote
    return Quote(source=PC.SYNC_SOURCE, symbol=C.INDEX).history(start=PC.SYNC_INIT_START, end=datetime.now().strftime("%Y-%m-%d"), interval=PC.SYNC_INTERVAL)


def sync_index(ext_dir=C.EXT_DIR, fetch=_fetch_index):
    """Write the index file from the source, keeping every bar with a positive close.

    Not through sync_stock_parquet: its OHLC-order check drops index bars whose high/low the source reports
    inconsistently (3 sessions of VNINDEX on 2026-11-10), and one missing session costs 60 windows of every ticker.
    Only the index close is used downstream."""
    raw = fetch()
    d = raw[list(PC.RAW_COLS)].copy()
    d = d[pd.to_numeric(d["close"], errors="coerce") > 0]
    t = pd.to_datetime(d[PC.TIME_COL])
    t = t.dt.tz_convert(None) if t.dt.tz is not None else t
    d[PC.TIME_COL] = (t.dt.normalize() + pd.Timedelta(hours=PC.SESSION_HOUR_UTC)).dt.tz_localize("UTC")
    d = d.drop_duplicates(PC.TIME_COL, keep="last").sort_values(PC.TIME_COL)
    path = Path(ext_dir) / PC.STOCK_FILE.format(ticker=C.INDEX)
    tmp = path.with_name(path.name + ".tmp")
    try:
        d.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


@dataclass
class Aux:
    frames: dict             # external symbol -> OHLCV on the calendar
    index: pd.DataFrame      # VNINDEX on the calendar
    glob: pd.DataFrame       # long table series, date, close
    aux_id: str | None


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def aux_id_of(d):
    d = Path(d)
    return _sha(f"{json.loads((d / SN.MANIFEST).read_text())['dataset_id']}:{_sha((d / GLOBAL).read_bytes())}".encode())


def freeze(ext_dir=C.EXT_DIR, global_file=C.GLOBAL_FILE, out_root=C.AUX):
    """Copy the stock files and the global table into out_root/<aux_id[:16]> with a manifest built from the copies."""
    out_root = Path(out_root)
    tmp = out_root / ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        for p in sorted(Path(ext_dir).glob(PC.STOCK_FILE.format(ticker="*"))):
            shutil.copy2(p, tmp)
        shutil.copy2(global_file, tmp / GLOBAL)
        SN.write_manifest(tmp, SN.build_manifest(tmp, PC.SYNC_SOURCE, datetime.now(timezone.utc).isoformat(timespec="seconds"), fetched=True))
        aid = aux_id_of(tmp)
        dest = out_root / aid[:16]
        if dest.exists():
            _verify(dest)
            if aux_id_of(dest) != aid:
                raise ValueError(f"{dest} đã tồn tại với nội dung khác")
            return dest
        os.replace(tmp, dest)
        return dest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _verify(d):
    d = Path(d)
    try:
        SN._check_files(d, json.loads((d / SN.MANIFEST).read_text()), "aux")
    except ValueError as e:
        raise ValueError(f"{d}: aux bị sửa sau khi đóng băng: {e}") from e
    if aux_id_of(d)[:16] != d.name:
        raise ValueError(f"{d}: aux bị sửa sau khi đóng băng (aux_id {aux_id_of(d)[:16]})")


def load(aux_dir, calendar):
    """Aux of a frozen directory on the price panel's calendar. Refuses a tampered directory or one older than the calendar."""
    d = Path(aux_dir)
    _verify(d)
    index = load_index(d, calendar)
    if not np.isfinite(index["close"].iloc[-1]):
        raise ValueError(f"{d}: aux cũ hơn dữ liệu giá ({C.INDEX} không có phiên {calendar[-1].date()})")
    return Aux(load_frames(d, calendar), index, pd.read_parquet(d / GLOBAL), aux_id_of(d))


if __name__ == "__main__":
    if sys.argv[1:] == ["index"]:
        print(sync_index())
    elif sys.argv[1:] == ["freeze"]:
        print(freeze())
    else:
        sys.exit("dùng: python -m ai_core.models.mmf.aux index | freeze")
