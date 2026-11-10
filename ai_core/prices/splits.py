"""Chia mẫu theo ngày thành bốn khối train/tune/calib/eval, có purge theo label_end_date.

Mốc fold cố định trong config.DATE_FOLDS (không tính lại lúc chạy, để dữ liệu mới append không dời mốc).
Mẫu thuộc khối khi as_of_session nằm trong khoảng khối VÀ label_end_date <= cuối khối (NaT bị loại).
"""
from typing import NamedTuple

import pandas as pd

from . import config as C

BLOCKS = ("train", "tune", "calib", "eval")


class Fold(NamedTuple):
    name: str
    train_end: pd.Timestamp
    tune_end: pd.Timestamp
    calib_end: pd.Timestamp
    eval_end: pd.Timestamp


def make_date_folds(calendar: pd.DatetimeIndex, last: str, block: int = C.BLOCK_SESSIONS, n: int = 3) -> list[Fold]:
    last = pd.Timestamp(last)
    if last not in calendar:
        raise ValueError(f"{last.date()} không phải phiên giao dịch trong lịch")
    pos = calendar.get_loc(last)
    if pos - (n + 2) * block < 0:
        raise ValueError(f"lịch quá ngắn cho {n} fold x {block} phiên (last ở vị trí {pos})")
    folds = [Fold(chr(ord("A") + n - 1 - j), *(calendar[pos - (j + k) * block] for k in (3, 2, 1, 0))) for j in range(n)]
    return folds[::-1]


def folds_from_config() -> list[Fold]:
    return [Fold(name, *map(pd.Timestamp, ends)) for name, *ends in C.DATE_FOLDS]


def _ranges(fold: Fold):
    """(tên khối, cận dưới loại trừ hoặc None, cuối khối)."""
    ends = fold[1:]
    return [(b, None if i == 0 else ends[i - 1], ends[i]) for i, b in enumerate(BLOCKS)]


def _masks(samples: pd.DataFrame, start, end):
    inside = samples["as_of_session"] <= end
    if start is not None:
        inside &= samples["as_of_session"] > start
    return inside, inside & (samples["label_end_date"] <= end)  # so với NaT cho False


def split_samples(samples: pd.DataFrame, fold: Fold) -> dict[str, pd.DataFrame]:
    return {b: samples[_masks(samples, s, e)[1]] for b, s, e in _ranges(fold)}


def split_manifest(samples: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    rows = []
    for f in folds:
        for b, s, e in _ranges(f):
            inside, kept = _masks(samples, s, e)
            rows.append((f.name, b, pd.NaT if s is None else s, e, int(kept.sum()), int((inside & ~kept).sum())))
    return pd.DataFrame(rows, columns=["fold", "block", "start", "end", "n_kept", "n_purged"])
