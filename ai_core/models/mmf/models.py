"""Branch encoders (price, text, fundamentals) and the concat/attention fusion network."""
import torch
from torch import nn
from . import config as C


class PriceEncoder(nn.Module):
    """LSTM/GRU last state, or a one-layer transformer last token over the C.LP window."""
    def __init__(self, f_in, kind):
        super().__init__()
        self.kind = kind
        if kind == "transformer":
            self.proj = nn.Linear(f_in, C.D); self.pos = nn.Parameter(torch.zeros(1, C.LP, C.D))
            self.enc = nn.TransformerEncoderLayer(C.D, nhead=4, dim_feedforward=64, dropout=C.DROPOUT, batch_first=True)
        else:
            self.rnn = (nn.LSTM if kind == "lstm" else nn.GRU)(f_in, C.D, batch_first=True)

    def forward(self, x):
        if self.kind == "transformer":
            return self.enc(self.proj(x) + self.pos)[:, -1]
        return self.rnn(x)[0][:, -1]


class GRUEncoder(nn.Module):
    """One-layer GRU, last state."""
    def __init__(self, f_in):
        super().__init__(); self.rnn = nn.GRU(f_in, C.D, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0][:, -1]


class FusionNet(nn.Module):
    """Logits (B, 3) from the arm's branches; last_weights (B, K) after an attention pass, else None."""
    def __init__(self, branches, fusion, n_tickers, f_price, f_text, f_fund, price_kind="lstm"):
        super().__init__()
        self.branches, self.fusion, self.last_weights = tuple(branches), fusion, None
        enc = {"price": lambda: PriceEncoder(f_price, price_kind), "text": lambda: GRUEncoder(f_text),
               "fund": lambda: nn.Sequential(nn.Linear(f_fund, C.D), nn.ReLU(), nn.Dropout(C.DROPOUT))}
        self.enc = nn.ModuleDict({b: enc[b]() for b in self.branches})
        if fusion == "attention":
            self.query = nn.Parameter(torch.randn(1, 1, C.D) * C.D ** -0.5)
            self.attn = nn.MultiheadAttention(C.D, 1, batch_first=True)
        width = C.D * len(self.branches) if fusion == "concat" else C.D
        self.ticker = nn.Embedding(n_tickers, C.EMB)
        self.drop = nn.Dropout(C.DROPOUT); self.out = nn.Linear(width + C.EMB, 3)

    def forward(self, x_price, x_text, x_fund, ticker):
        x = {"price": x_price, "text": x_text, "fund": x_fund}
        z = torch.stack([self.enc[b](x[b]) for b in self.branches], 1)      # (B, K, D)
        if self.fusion == "attention":
            z, w = self.attn(self.query.expand(len(z), -1, -1), z, z)
            self.last_weights = w[:, 0]                                      # (B, 1, K) -> (B, K)
        z = z.flatten(1) if self.fusion == "concat" else z[:, 0]
        return self.out(self.drop(torch.cat([z, self.ticker(ticker)], 1)))
