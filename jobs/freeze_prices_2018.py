"""Frozen long-history dataset for experiments: data/prices_2018 put on one common date axis and cut at the last session
of the dev snapshot, written with a manifest to data/prices_2018/snapshots/<dataset_id[:16]>.

A session a ticker has no bar for (vendor gaps in 2019, ACB's move from HNX to HOSE in 12/2020) becomes a NaN row, never a
filled one: features and windows touching it drop out downstream. From 2021-01-04 on the rows ARE the dev snapshot's; the
older rows are rescaled onto its adjustment basis (the vendor re-adjusts a whole series after an ex-date: VNM, 11/2026).
Use: .venv/bin/python jobs/freeze_prices_2018.py
"""
from datetime import datetime, timezone

import pandas as pd

from ai_core import paths
from ai_core.prices import config as C
from ai_core.prices.snapshot import build_manifest, snapshot_dir, write_manifest

SRC = paths.DATA / "prices_2018"
DEV = paths.PRICES / "snapshots/24251acc1a5695bd"


def main():
    raw = {p.name.split("_")[0]: pd.read_parquet(p)[list(C.RAW_COLS)].set_index(C.TIME_COL).sort_index()
           for p in sorted(SRC.glob(C.STOCK_FILE.format(ticker="*")))}
    end = max(pd.read_parquet(p, columns=[C.TIME_COL])[C.TIME_COL].max() for p in DEV.glob("*.parquet"))
    axis = sorted(set().union(*(d.index for d in raw.values())))
    axis = pd.DatetimeIndex([t for t in axis if t <= end], name=C.TIME_COL)
    tmp = SRC / "snapshots/.tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for sym, d in raw.items():
        out = d.reindex(axis).astype(float)
        dev = pd.read_parquet(DEV / C.STOCK_FILE.format(ticker=sym)).set_index(C.TIME_COL)[list(out.columns)].astype(float)
        ratio = dev["close"] / out.loc[dev.index, "close"]
        k = float(ratio.median())
        if (ratio / k - 1).abs().max() > 1e-3 or not dev[C.VOLUME_COL].equals(out.loc[dev.index, C.VOLUME_COL]):
            raise SystemExit(f"{sym}: differs from the dev snapshot by more than one constant price factor")
        old = out.index < dev.index[0]
        out.loc[old, list(C.PRICE_COLS)] = (out.loc[old, list(C.PRICE_COLS)] * k).round(2)
        out.loc[dev.index] = dev
        out.reset_index().to_parquet(tmp / C.STOCK_FILE.format(ticker=sym), index=False)
        print(f"{sym}: {len(out)} rows, {int(out['close'].isna().sum())} without a bar, basis factor {k:.4f}")
    m = build_manifest(tmp, "kbs (data/prices_2018, aligned)", datetime.now(timezone.utc).isoformat(timespec="seconds"), False)
    write_manifest(tmp, m)
    snap = snapshot_dir(SRC, m["dataset_id"])
    if snap.exists():
        raise SystemExit(f"{snap} already exists")
    tmp.rename(snap)
    print(snap)


if __name__ == "__main__":
    main()
