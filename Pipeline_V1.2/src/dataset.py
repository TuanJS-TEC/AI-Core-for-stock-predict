"""Dataset giá dùng chung cho train và inference: cả hai đi qua _features nên đặc trưng giống hệt."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel
from .labels import make_labels
from .price_features import FEATURES, compute, market_features

LABEL_COLS = ["target_session", "label_end_date", "realized_return", "theta_used", "label", "status", "label_version"]
_FEAT_COLS = [*FEATURES, "flag_hl_equal"]


def _features(panel: Panel) -> dict[str, pd.DataFrame]:
    """Đặc trưng theo mã trên lịch phiên (không điền, không bỏ dòng)."""
    if panel.ohlcv is None:
        panel.on_calendar("")  # ném đúng ValueError của Panel
    frames = {s: panel.on_calendar(s) for s in sorted(panel.ohlcv)}
    market = market_features(frames)
    return {s: compute(f, market) for s, f in frames.items()}


def _long(feats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat(feats, names=["symbol", "as_of_session"]).reset_index()
    out["features_ok"] = np.isfinite(out[FEATURES].to_numpy(dtype=float)).all(axis=1)
    return out


def build_dataset(panel: Panel, h: int) -> pd.DataFrame:
    feats = _features(panel)
    labels = {s: make_labels(panel.on_calendar(s), h) for s in feats}
    out = _long({s: f.join(labels[s]) for s, f in feats.items()})
    out["h"] = int(h)
    return out[["symbol", "as_of_session", "h", *_FEAT_COLS, "features_ok", *LABEL_COLS]]


def inference_features(panel: Panel) -> pd.DataFrame:
    feats = {s: f.iloc[[-1]] for s, f in _features(panel).items()}  # f is indexed by panel.calendar
    return _long(feats)[["symbol", "as_of_session", *_FEAT_COLS, "features_ok"]]
