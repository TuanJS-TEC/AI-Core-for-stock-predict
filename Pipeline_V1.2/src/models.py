"""Linear, NLinear, DLinear và mô hình gộp 15 mã: đầu hồi quy + đầu phân loại có embedding mã."""
import torch
from torch import nn

from . import config as C


class Linear(nn.Module):
    def __init__(self, L: int, T: int):
        super().__init__()
        self.fc = nn.Linear(L, T)

    def forward(self, x):  # x: (B, L)
        return self.fc(x)


class NLinear(nn.Module):
    """Trừ giá trị cuối của cửa sổ, qua Linear, rồi cộng lại giá trị cuối."""

    def __init__(self, L: int, T: int):
        super().__init__()
        self.fc = nn.Linear(L, T)

    def forward(self, x):
        last = x[:, -1:]
        return self.fc(x - last) + last


class MovingAvg(nn.Module):
    """Moving average kernel cố định, padding hai đầu bằng giá trị biên để giữ độ dài."""

    def __init__(self, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.pool = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)

    def forward(self, x):  # (B, L)
        pad = (self.kernel - 1) // 2
        front = x[:, :1].repeat(1, pad)
        end = x[:, -1:].repeat(1, pad)
        z = torch.cat([front, x, end], dim=1).unsqueeze(1)
        return self.pool(z).squeeze(1)


class DLinear(nn.Module):
    def __init__(self, L: int, T: int, kernel: int = C.MA_KERNEL):
        super().__init__()
        self.avg = MovingAvg(kernel)
        self.fc_trend = nn.Linear(L, T)
        self.fc_season = nn.Linear(L, T)

    def forward(self, x):
        trend = self.avg(x)
        season = x - trend
        return self.fc_trend(trend) + self.fc_season(season)


REGISTRY = {"Linear": Linear, "NLinear": NLinear, "DLinear": DLinear}


def arch_spec() -> dict:
    """Chữ ký kiến trúc: kết quả huấn luyện và model đã lưu chỉ dùng lại được khi chữ ký này không đổi."""
    return {"input": C.TARGET_COL, "cls_head": "embed", "cls_temporal_dim": C.CLS_TEMPORAL_DIM,
            "ticker_emb_dim": C.TICKER_EMB_DIM}


def build_model(name: str, L: int, T: int) -> nn.Module:
    return REGISTRY[name](L, T)


class FeatureProjection(nn.Module):
    """Chiếu thời gian L -> d riêng cho từng kênh: (B, F, L) -> (B, F, d). Khởi tạo như nn.Linear."""

    def __init__(self, n_feat: int, L: int, d: int):
        super().__init__()
        bound = 1 / L ** 0.5
        self.weight = nn.Parameter(torch.empty(n_feat, L, d).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(n_feat, d).uniform_(-bound, bound))

    def forward(self, x):
        return torch.einsum("bfl,fld->bfd", x, self.weight) + self.bias


class EmbedClsHead(nn.Module):
    """Đầu phân loại có embedding mã (phương án C, plan_pooled_features.md mục 1.4 và 2.3).

    Tầng 1, chiếu thời gian L -> d, cùng họ kiến trúc với mô hình hồi quy:
      Linear   proj(x)
      NLinear  proj(x - x_cuối), rồi nối thêm x_cuối vào tầng 2 để giữ mức hiện tại của chuỗi
      DLinear  proj_trend(trend) + proj_season(x - trend), moving average kernel C.MA_KERNEL, padding hai đầu
    Tầng 2: nối (F·d) với embedding của mã, Linear -> N_CLASSES. Embedding cho mỗi mã một độ dịch riêng trên
    không gian đặc trưng trong khi phần chiếu thời gian học chung từ mẫu của cả 15 mã.
    """

    def __init__(self, name: str, L: int, n_feat: int, n_tickers: int, d: int = C.CLS_TEMPORAL_DIM,
                 emb_dim: int = C.TICKER_EMB_DIM, kernel: int = C.MA_KERNEL, n_classes: int = C.N_CLASSES):
        super().__init__()
        self.name = name
        if name == "DLinear":
            self.avg = MovingAvg(kernel)
            self.proj_trend = FeatureProjection(n_feat, L, d)
            self.proj_season = FeatureProjection(n_feat, L, d)
        else:
            self.proj = FeatureProjection(n_feat, L, d)
        width = n_feat * d + (n_feat if name == "NLinear" else 0)
        self.emb = nn.Embedding(n_tickers, emb_dim)
        self.mix = nn.Linear(width + emb_dim, n_classes)

    def forward(self, x, tid):  # x: (B, F, L), tid: (B,) -> (B, n_classes)
        b, f, length = x.shape
        if self.name == "DLinear":
            z = x.reshape(b * f, length)
            trend = self.avg(z).reshape(b, f, length)
            h = self.proj_trend(trend) + self.proj_season(x - trend)
        elif self.name == "NLinear":
            last = x[:, :, -1:]
            h = torch.cat([self.proj(x - last).reshape(b, -1), last.reshape(b, f)], dim=1)
        else:
            h = self.proj(x)
        return self.mix(torch.cat([h.reshape(b, -1), self.emb(tid)], dim=1))


class PooledMultiTask(nn.Module):
    """Mô hình gộp 15 mã. Đầu hồi quy (Linear/NLinear/DLinear) trên kênh 0 và đầu phân loại EmbedClsHead,
    hai đầu không chia sẻ tham số.

    Đầu vào x (B, F, L), tid (B,) -> (hồi quy (B, T), logits (B, N_CLASSES)).
    """

    def __init__(self, name: str, L: int, T: int, n_feat: int, n_tickers: int):
        super().__init__()
        self.reg = build_model(name, L, T)   # tạo trước đầu phân loại: giữ thứ tự khởi tạo của các lần chạy trước
        self.cls = EmbedClsHead(name, L, n_feat, n_tickers)

    def forward(self, x, tid):
        return self.reg(x[:, 0]), self.cls(x, tid)
