"""Manifest, dataset_id và snapshot bất biến của dữ liệu giá ngày.

- content_hash: sha256 của CSV chuẩn hóa (cột RAW_COLS, sắp theo time, time ISO UTC, float %.6f); không phụ thuộc
  thứ tự dòng hay cách parquet mã hóa, nhưng đổi một giá trị OHLC là đổi hash.
- dataset_id: sha256 của các dòng `SYM:content_sha256` sắp theo mã; không phụ thuộc fetched_at.
- freeze: sao parquet + manifest vào stock_dir/snapshots/<dataset_id[:16]> để chạy lại đúng dữ liệu đã dùng.

Dùng: python -m src.snapshot freeze     # in dataset_id của manifest hiện tại
"""
import hashlib
import importlib.metadata as md
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .validation import load_holidays, unexplained_gaps

MANIFEST = "manifest.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(df: pd.DataFrame) -> str:
    out = df[list(C.RAW_COLS)].copy()
    out[C.TIME_COL] = pd.to_datetime(out[C.TIME_COL], utc=True)     # ponytail: time không múi giờ coi là UTC
    out = out.sort_values(C.TIME_COL)
    out[C.TIME_COL] = out[C.TIME_COL].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return _sha256(out.to_csv(index=False, float_format="%.6f").encode())


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _holidays_unverified() -> int:
    return 0 if not Path(C.HOLIDAYS_FILE).exists() else int((~pd.read_csv(C.HOLIDAYS_FILE)["verified"].astype(bool)).sum())


def build_manifest(stock_dir: Path, source: str, fetched_at: str, fetched: bool,
                   holidays: np.ndarray | None = None) -> dict:
    holidays = load_holidays() if holidays is None else holidays
    tickers, quality = {}, {}
    for p in sorted(Path(stock_dir).glob(C.STOCK_FILE.format(ticker="*"))):
        sym = p.name.split("_")[0]
        df = pd.read_parquet(p)
        t = pd.to_datetime(df[C.TIME_COL]).sort_values()
        tickers[sym] = {"rows": len(df), "first": t.iloc[0].isoformat(), "last": t.iloc[-1].isoformat(),
                        "content_sha256": content_hash(df), "file_sha256": _sha256(p.read_bytes())}
        days = pd.DatetimeIndex(t.dt.tz_localize(None).dt.normalize()) if t.dt.tz is not None else \
            pd.DatetimeIndex(t.dt.normalize())
        quality[sym] = {"unexplained_gaps": [d.date().isoformat() for d in unexplained_gaps(days, holidays)]}
    ids = "\n".join(f"{s}:{tickers[s]['content_sha256']}" for s in sorted(tickers))
    return {"dataset_id": _sha256(ids.encode()), "fetched_at": fetched_at, "fetched": fetched, "source": source,
            "adjustment_policy": "unknown",
            "libs": {k: _version(k) for k in ("vnstock", "pandas", "pyarrow")},
            "tickers": tickers, "quality": quality, "holidays_unverified": _holidays_unverified()}


def write_manifest(stock_dir: Path, manifest: dict) -> None:
    path = Path(stock_dir) / MANIFEST
    tmp = path.with_name(MANIFEST + ".tmp")
    try:
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def snapshot_dir(stock_dir: Path, dataset_id: str) -> Path:
    return Path(stock_dir) / "snapshots" / dataset_id[:16]


def _check_files(d: Path, manifest: dict, where: str) -> None:
    for sym, info in manifest["tickers"].items():
        p = d / C.STOCK_FILE.format(ticker=sym)
        if not p.exists() or _sha256(p.read_bytes()) != info["file_sha256"]:
            raise ValueError(f"{where}: {p.name} không khớp manifest (dataset_id {manifest['dataset_id'][:16]})")


def freeze(stock_dir: Path) -> str:
    stock_dir = Path(stock_dir)
    manifest = json.loads((stock_dir / MANIFEST).read_text())
    snap = snapshot_dir(stock_dir, manifest["dataset_id"])
    if snap.exists():
        _check_files(snap, manifest, "snapshot")
        return manifest["dataset_id"]
    _check_files(stock_dir, manifest, "dữ liệu sống")                # manifest cũ so với parquet đã đổi: không đóng băng
    tmp = snap.with_name(snap.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for sym in manifest["tickers"]:
        shutil.copy2(stock_dir / C.STOCK_FILE.format(ticker=sym), tmp)
    shutil.copy2(stock_dir / MANIFEST, tmp)
    os.replace(tmp, snap)
    return manifest["dataset_id"]


if __name__ == "__main__":
    if sys.argv[1:] != ["freeze"]:
        sys.exit("dùng: python -m src.snapshot freeze")
    print(freeze(C.STOCK_DIR))
