"""Dữ liệu gộp 15 mã: gộp mẫu, không gộp chuỗi (plan_pooled_features.md, câu hỏi 1).

- Đầu vào duy nhất là log giá đóng cửa. Mọi mã phải có cùng dãy ngày (sync_stock_parquet.py bảo đảm mỗi phiên
  một dòng); các mã được căn theo vị trí dòng.
- Chia 70:15:15 theo thời gian trên trục ngày chung: split_sizes(n) cho mọi mã (n đổi khi sync dữ liệu mới), nên
  mục tiêu của mọi cửa sổ train (ở mọi mã) nằm trước ngày đầu tiên của val. Đầu vào val/test lấy L điểm liền trước.
- Cửa sổ được cắt bên trong chuỗi của từng mã, rồi mới nối các mẫu của mọi mã. Mỗi mẫu mang ticker_id
  (dùng cho embedding mã), end_idx (chỉ số dòng của điểm cuối cửa sổ) và ngày mục tiêu.
- Scaler fit riêng từng mã, chỉ trên các dòng train của mã đó.
- r = log-return thật giữa bước T và điểm cuối cửa sổ; θ = K_THETA · std(r) trên mọi cửa sổ train của mã;
  nhãn 0 Giảm / 1 Đi ngang / 2 Tăng theo ±θ.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from .metrics import class_prior


def split_sizes(n: int) -> tuple[int, int, int]:
    n_train = int(n * C.TRAIN_RATIO)
    n_val = int(n * C.VAL_RATIO)
    return n_train, n_val, n - n_train - n_val


def assign_labels(r: np.ndarray, theta: float) -> np.ndarray:
    """0 = Giảm, 1 = Đi ngang, 2 = Tăng."""
    lab = np.full(len(r), C.CLASS_FLAT, dtype=np.int64)
    lab[r > theta] = C.CLASS_UP
    lab[r < -theta] = C.CLASS_DOWN
    return lab


def raw_windows(series: np.ndarray, end_idx: np.ndarray, L: int) -> np.ndarray:
    """series: (n,). Cửa sổ L dòng liền nhau kết thúc tại từng end_idx: (N, L)."""
    return series[np.asarray(end_idx)[:, None] - L + 1 + np.arange(L)]


def make_windows(series: np.ndarray, start: int, stop: int, L: int, T: int):
    """series: (n, C). Trả về x (N, C, L), y (N, C, T). Mục tiêu nằm trong [start, stop)."""
    n_win = stop - start - T + 1
    idx = np.arange(n_win)[:, None]
    x = series[start - L + idx + np.arange(L)]          # (N, L, C)
    y = series[start + idx + np.arange(T)]              # (N, T, C)
    return x.transpose(0, 2, 1), y.transpose(0, 2, 1)


# ---------------------------------------------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------------------------------------------
def stock_path(ticker: str, stock_dir=None):
    return Path(C.STOCK_DIR if stock_dir is None else stock_dir) / C.STOCK_FILE.format(ticker=ticker)


def load_ohlcv(ticker: str, stock_dir=None) -> pd.DataFrame:
    """OHLCV của một mã, chỉ số là ngày (đã bỏ giờ) để căn các mã, cột stamp giữ thời điểm gốc (UTC, không múi giờ)."""
    df = pd.read_parquet(stock_path(ticker, stock_dir), columns=list(C.RAW_COLS)).sort_values(C.TIME_COL)
    t = pd.to_datetime(df[C.TIME_COL])
    if t.dt.tz is not None:
        t = t.dt.tz_convert(None)
    df["stamp"] = t                              # thời điểm gốc, duy nhất theo dòng, dùng làm cột date đầu ra
    df.index = pd.DatetimeIndex(t.dt.normalize(), name="date")
    return df.drop(columns=C.TIME_COL)


def load_raw(tickers=C.TICKERS, stock_dir=None) -> dict[str, pd.DataFrame]:
    return {t: load_ohlcv(t, stock_dir) for t in tickers}


def check_aligned(raw: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Mọi mã phải có đúng cùng dãy ngày theo từng dòng, mỗi phiên một dòng."""
    first = next(iter(raw))
    ref = raw[first].index
    for t, df in raw.items():
        if not df.index.equals(ref) or not df["stamp"].equals(raw[first]["stamp"]):
            raise ValueError(f"{t}: dãy ngày khác mã {first}, không gộp theo ngày được")
    dup = ref[ref.duplicated()]
    if len(dup):
        raise ValueError(f"Ngày lặp {[str(d.date()) for d in dup]}; chạy `python sync_stock_parquet.py --no-fetch` "
                         "để làm sạch trước")
    return ref


@dataclass
class Panel:
    dates: pd.DatetimeIndex              # trục ngày chung của mọi mã (đã bỏ giờ)
    stamps: np.ndarray                   # thời điểm gốc từng dòng (datetime64, duy nhất), dùng làm ngày đầu ra
    log_close: dict[str, np.ndarray]     # mã -> (n,)
    dataset_id: str | None = None        # từ stock_dir/manifest.json; None nếu thư mục chưa có manifest


def build_panel(tickers=C.TICKERS, raw: dict[str, pd.DataFrame] | None = None, stock_dir=None) -> Panel:
    raw = load_raw(tickers, stock_dir) if raw is None else raw
    dates = check_aligned(raw)
    stamps = raw[next(iter(raw))]["stamp"].to_numpy()
    manifest = Path(C.STOCK_DIR if stock_dir is None else stock_dir) / "manifest.json"
    dataset_id = json.loads(manifest.read_text())["dataset_id"] if manifest.exists() else None
    return Panel(dates, stamps, {t: np.log(raw[t][C.TARGET_COL].to_numpy(np.float64)) for t in tickers}, dataset_id)


# ---------------------------------------------------------------------------------------------------------------
# Tập gộp
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class PooledSplit:
    x: np.ndarray         # (N, 1, L) log giá đã chuẩn hóa theo từng mã
    y: np.ndarray         # (N, T) giá trị chuẩn hóa tương lai (mục tiêu hồi quy)
    labels: np.ndarray    # (N,) nhãn thật 0 Giảm / 1 Đi ngang / 2 Tăng
    r: np.ndarray         # (N,) log-return thật giữa điểm cuối cửa sổ và bước T
    ticker: np.ndarray    # (N,) chỉ số mã trong PooledData.tickers
    end_idx: np.ndarray   # (N,) chỉ số dòng của điểm cuối cửa sổ (trong chuỗi của mã đó)
    date: np.ndarray      # (N,) thời điểm mục tiêu (bước T), datetime64

    def of(self, k: int) -> np.ndarray:
        """Mặt nạ các mẫu của mã thứ k."""
        return self.ticker == k


@dataclass
class PooledData:
    L: int
    T: int
    tickers: list[str]
    sizes: tuple[int, int, int]
    scalers: dict[str, StandardScaler]    # mã -> scaler của log close
    theta: dict[str, float]
    prior: dict[str, np.ndarray]          # tần suất lớp ở train của từng mã
    train: PooledSplit
    val: PooledSplit
    test: PooledSplit

    @property
    def n_feat(self) -> int:
        return self.train.x.shape[1]


def build_pooled(L: int, T: int, tickers=C.TICKERS, panel: Panel | None = None) -> PooledData:
    panel = build_panel(tickers) if panel is None else panel
    tickers = list(tickers)
    n = len(panel.dates)
    n_train, n_val, n_test = split_sizes(n)
    if L >= n_train:
        raise ValueError(f"L {L} không còn cửa sổ train")
    bounds = {"train": (L, n_train), "val": (n_train, n_train + n_val),
              "test": (n_train + n_val, n_train + n_val + n_test)}

    parts = {k: [] for k in bounds}
    scalers, theta, prior = {}, {}, {}
    for k, t in enumerate(tickers):
        logc = panel.log_close[t][:, None]
        scaler = StandardScaler().fit(logc[:n_train])                 # chỉ dòng train của mã này
        scaled = scaler.transform(logc)
        rs = {}
        for name, (a, b) in bounds.items():
            xr, yr = make_windows(logc, a, b, L, T)
            rs[name] = yr[:, 0, -1] - xr[:, 0, -1]
        theta[t] = float(C.K_THETA * rs["train"].std())                # mọi cửa sổ train của mã
        for name, (a, b) in bounds.items():
            x, y = make_windows(scaled, a, b, L, T)
            end = a + np.arange(len(x)) - 1
            parts[name].append(dict(x=x, y=y[:, 0], labels=assign_labels(rs[name], theta[t]), r=rs[name],
                                    ticker=np.full(len(x), k), end_idx=end, date=panel.stamps[end + T]))
        scalers[t] = scaler
        prior[t] = class_prior(parts["train"][-1]["labels"])
    splits = {name: PooledSplit(**{f: np.concatenate([p[f] for p in ps]) for f in ps[0]})
              for name, ps in parts.items()}
    for s in splits.values():
        s.x = s.x.astype(np.float32)
        s.y = s.y.astype(np.float32)
    return PooledData(L, T, tickers, (n_train, n_val, n_test), scalers, theta, prior,
                      splits["train"], splits["val"], splits["test"])


def make_loader(split: PooledSplit, shuffle: bool, seed: int, batch_size: int = C.BATCH_SIZE) -> DataLoader:
    """Chỉ train được shuffle (trộn cửa sổ của mọi mã và mọi ngày)."""
    ds = TensorDataset(torch.from_numpy(split.x), torch.from_numpy(split.y), torch.from_numpy(split.labels).long(),
                       torch.from_numpy(split.ticker).long())
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)
